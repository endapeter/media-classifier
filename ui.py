"""
Aesthetic UI layer.

Visual-only: this styles console output using ANSI truecolor escapes around
a palette centered on #de1951. It intentionally fails safe: if styling
cannot be applied, output falls back to plain printing.

Modules throughout the tool do `from ui import print` so that every log
line passes through the styled printer.
"""

import os
import sys
import math
import re
import time
from typing import Optional, Tuple

UI_ACCENT_HEX = "#de1951"

UI_ACCENT = (222, 25, 81)
UI_ACCENT_DARK = (84, 8, 36)
UI_ACCENT_DEEP = (132, 14, 52)
UI_ACCENT_LIGHT = (255, 84, 128)
UI_ACCENT_SOFT = (255, 151, 178)

UI_TEXT = (236, 236, 241)
UI_MUTED = (157, 157, 168)
UI_WARNING = (255, 143, 107)
UI_ERROR = (255, 64, 88)
UI_SUCCESS = UI_ACCENT_LIGHT

UI_FORCE_COLOR = os.environ.get("UI_FORCE_COLOR", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
UI_DISABLE_COLOR = os.environ.get("UI_DISABLE_COLOR", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
UI_ANIMATIONS = os.environ.get("UI_NO_ANIMATION", "").strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}

try:
    UI_INTRO_DURATION = max(0.0, float(os.environ.get("UI_INTRO_DURATION", "0.45")))
except Exception:
    UI_INTRO_DURATION = 0.45

_PLAIN_PRINT = print


def _ui_enable_windows_ansi() -> None:
    """Enable ANSI escape processing on modern Windows consoles."""
    if os.name != "nt":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()

        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(
                handle,
                mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
            )
    except Exception:
        pass


_ui_enable_windows_ansi()


def _ui_is_tty(stream=None) -> bool:
    stream = stream if stream is not None else sys.stdout
    try:
        return stream.isatty()
    except Exception:
        return False


def _ui_color_enabled(stream=None) -> bool:
    if UI_DISABLE_COLOR:
        return False

    if UI_FORCE_COLOR:
        return True

    if os.environ.get("NO_COLOR"):
        return False

    if os.environ.get("TERM") == "dumb":
        return False

    return _ui_is_tty(stream)


def _ui_can_animate(stream=None) -> bool:
    return UI_ANIMATIONS and _ui_is_tty(stream) and _ui_color_enabled(stream)


def _ui_paint(text: str, rgb: Tuple[int, int, int], bold: bool = False) -> str:
    if not text:
        return ""

    r, g, b = rgb
    prefix = "1;38" if bold else "38"
    return f"\033[{prefix};2;{r};{g};{b}m{text}\033[0m"


def _ui_blend(
    a: Tuple[int, int, int],
    b: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


_UI_GRADIENT_STOPS = [
    UI_ACCENT_DARK,
    UI_ACCENT_DEEP,
    UI_ACCENT,
    UI_ACCENT_LIGHT,
    UI_ACCENT_SOFT,
    UI_ACCENT,
]


def _ui_gradient_color(t: float) -> Tuple[int, int, int]:
    """Sample a cyclic gradient centered around #de1951."""
    t = t % 1.0
    stops = _UI_GRADIENT_STOPS + [_UI_GRADIENT_STOPS[0]]
    scaled = t * (len(stops) - 1)
    idx = int(scaled)
    frac = scaled - idx

    if idx >= len(stops) - 1:
        return stops[-1]

    return _ui_blend(stops[idx], stops[idx + 1], frac)


def _ui_gradient_text(text: str, shift: float = 0.0, bold: bool = True) -> str:
    """Render text using a moving-style gradient palette."""
    if not text:
        return text

    if not _ui_color_enabled(sys.stdout):
        return text

    try:
        out = []

        for i, ch in enumerate(text):
            if ch == " ":
                out.append(" ")
                continue

            color = _ui_gradient_color(shift + i * 0.035)
            out.append(_ui_paint(ch, color, bold=bold))

        return "".join(out)
    except Exception:
        return text


def _ui_blinky_dots(t: float) -> str:
    """Three pulsing dots in the accent palette."""
    try:
        parts = []

        for i in range(3):
            wave = math.sin(t * 6.0 - i * 0.9)
            brightness = 0.2 + 0.8 * max(0.0, wave)
            color = _ui_blend(UI_ACCENT_DARK, UI_ACCENT_LIGHT, brightness)
            parts.append(_ui_paint("● ", color, bold=False))

        return "".join(parts) + "\033[0m"
    except Exception:
        return ""


def _ui_animated_line(
    text: str,
    subtitle: Optional[str] = None,
    duration: Optional[float] = None,
    fps: int = 30,
) -> None:
    """
    Brief animated line with blinky dots and moving gradient.

    This is decorative only. If animation is unavailable, it prints a static
    styled line instead.
    """
    if duration is None:
        duration = UI_INTRO_DURATION

    stream = sys.stdout

    if not _ui_can_animate(stream) or duration <= 0:
        _PLAIN_PRINT(_ui_gradient_text(text, shift=0.25, bold=True))
        if subtitle:
            _PLAIN_PRINT(_ui_paint(subtitle, UI_MUTED))
        return

    frames = max(1, int(duration * fps))
    frame_time = 1.0 / max(1, fps)

    try:
        stream.write("\033[?25l")
        start = time.monotonic()
        next_tick = start

        for _ in range(frames):
            t = time.monotonic() - start
            line = (
                f"{_ui_blinky_dots(t)} "
                f"{_ui_gradient_text(text, shift=t * 0.85, bold=True)}"
            )

            stream.write(f"\r\033[2K{line}")
            stream.flush()

            next_tick += frame_time
            time.sleep(max(0.0, next_tick - time.monotonic()))

        stream.write("\r\033[2K")
        stream.flush()
    finally:
        try:
            stream.write("\033[?25h")
            stream.flush()
        except Exception:
            pass

    _PLAIN_PRINT(_ui_gradient_text(text, shift=0.25, bold=True))

    if subtitle:
        _PLAIN_PRINT(_ui_paint(subtitle, UI_MUTED))


def _ui_intro() -> None:
    if not _ui_color_enabled(sys.stdout):
        return

    _ui_animated_line(
        "Image Library Organizer",
        "Qwen2.5-VL • OpenVINO • content-aware organization",
        UI_INTRO_DURATION,
    )


_UI_LABEL_PREFIXES = (
    "Date token",
    "Event folder",
    "Event name",
    "Planned destination",
    "Processed/planned",
    "Skipped already processed",
    "Resumed from checkpoint",
    "Failed",
)


def _ui_style_line(line: str) -> str:
    """
    Style a single line of normal script output.

    This is intentionally conservative: it colors markers and headings, but
    does not change the text content.
    """
    if not line:
        return line

    # Never restyle already-styled output.
    if "\033" in line:
        return line

    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    body = stripped.rstrip()

    if not body:
        return line

    low = body.lower()

    if body.startswith("[Error]"):
        return indent + _ui_paint(body, UI_ERROR, bold=True)

    if body.startswith("[Warning]") or body.startswith("Notice:"):
        return indent + _ui_paint(body, UI_WARNING)

    if body.startswith("Summary:") or body.startswith("Video summary:"):
        return indent + _ui_gradient_text(body, shift=0.22, bold=True)

    if body.startswith("Dry run complete"):
        return indent + _ui_paint(body, UI_WARNING, bold=True)

    if body.startswith("Found") and (
        "images" in low
        or "videos" in low
        or "analysis checkpoint" in low
    ):
        return indent + _ui_paint(body, UI_ACCENT_LIGHT, bold=True)

    if body.startswith("Loading") or body.startswith("Model loaded"):
        return indent + _ui_paint(body, UI_ACCENT_LIGHT)

    if body.startswith("Merging"):
        return indent + _ui_paint(body, UI_WARNING)

    if body.startswith("Est."):
        return indent + _ui_paint(body, UI_MUTED)

    if body.startswith("Failed to load") or body.startswith("Failed to process"):
        return indent + _ui_paint(body, UI_ERROR)

    if body.startswith("Retrying") or body.startswith("Trying next device"):
        return indent + _ui_paint(body, UI_MUTED)

    if body.startswith("- "):
        return indent + _ui_paint(body, UI_MUTED)

    progress_match = re.match(r"^\[\d+/\d+\]", body)
    if progress_match:
        return indent + re.sub(
            r"^(\[\d+/\d+\])",
            lambda m: _ui_paint(m.group(1), UI_ACCENT_LIGHT, bold=True),
            body,
            count=1,
        )

    if "->" in body:
        return indent + body.replace(
            "->",
            _ui_paint("->", UI_SUCCESS, bold=True),
            1,
        )

    for label in _UI_LABEL_PREFIXES:
        if body.startswith(label + ":"):
            rest = body[len(label) + 1 :]

            rest_color = UI_TEXT
            if label == "Failed":
                try:
                    if int(rest.strip()) > 0:
                        rest_color = UI_ERROR
                except Exception:
                    pass

            return (
                indent
                + _ui_paint(label + ":", UI_MUTED)
                + _ui_paint(rest, rest_color)
            )

    if body.startswith("Already processed") or body.startswith("Analysis resumed"):
        return indent + _ui_paint(body, UI_MUTED)

    return line


def _ui_style_message(text: str, stream=None) -> str:
    if not isinstance(text, str):
        return text

    if not _ui_color_enabled(stream):
        return text

    try:
        return "\n".join(_ui_style_line(part) for part in text.split("\n"))
    except Exception:
        return text


def _ui_pretty_print(*args, **kwargs):
    """
    Drop-in print replacement that applies aesthetic styling only.

    If anything goes wrong while styling, it falls back to plain output so
    functionality is never compromised.
    """
    if not args:
        return _PLAIN_PRINT(*args, **kwargs)

    stream = kwargs.get("file", sys.stdout)

    if not _ui_color_enabled(stream):
        return _PLAIN_PRINT(*args, **kwargs)

    try:
        styled_args = []

        for arg in args:
            if isinstance(arg, str):
                styled_args.append(_ui_style_message(arg, stream))
            else:
                styled_args.append(arg)

        return _PLAIN_PRINT(*styled_args, **kwargs)
    except Exception:
        return _PLAIN_PRINT(*args, **kwargs)


# Styled drop-in for the builtin print. Other modules use
# `from ui import print` so their log lines get styled too.
print = _ui_pretty_print
