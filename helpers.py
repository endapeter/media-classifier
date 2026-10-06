"""
Small shared helpers: word/text cleanup, year validation, stable hashing,
and the time-estimate printer.
"""

import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List

from config import CURRENT_YEAR, GENERIC_NAME_WORDS, TIMING_WARMUP_IMAGES
from ui import print


def valid_year(value: Any) -> bool:
    """Return True if value looks like a plausible year."""
    try:
        y = int(str(value).strip())
    except Exception:
        return False

    return 1900 <= y <= CURRENT_YEAR + 1


def words_from_text(text: Any) -> List[str]:
    """
    Convert arbitrary text into clean lowercase words.

    Example:
        "Family / Beach Vacation!" -> ["family", "beach", "vacation"]
    """
    if text is None:
        return []

    text = str(text)

    # Convert path separators to spaces.
    text = re.sub(r"[\\/]+", " ", text)

    # Keep letters, digits, unicode words, spaces, hyphens.
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)

    # Normalize underscores, hyphens, and whitespace to single spaces.
    text = re.sub(r"[\s_\-]+", " ", text, flags=re.UNICODE).strip().lower()

    if not text:
        return []

    return text.split()


def remove_year_words(words: List[str]) -> List[str]:
    """Remove standalone year tokens like 2023 from theme words."""
    return [w for w in words if not re.fullmatch(r"(?:19|20)\d{2}", w)]


def remove_generic_words(words: List[str]) -> List[str]:
    """Remove content-free camera/filename tokens like 'img' or 'dsc'."""
    return [w for w in words if w not in GENERIC_NAME_WORDS]


def stable_hash(img_path: Path) -> str:
    """
    Short stable hash used to avoid filename collisions.
    Based on path, size, and modification time.
    """
    try:
        st = img_path.stat()
        payload = f"{img_path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    except Exception:
        payload = str(img_path).encode("utf-8")

    return hashlib.md5(payload).hexdigest()[:6]


def clean_existing_folder_part(part: str) -> str:
    """Make an existing folder name safer when recreating the path."""
    part = re.sub(r'[<>:"/\\|?*]+', "_", part).strip()
    part = part.rstrip(".")

    if not part:
        return "folder"

    return part


# ============================================================
# Time estimation
# ============================================================

def format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""
    seconds = max(0, int(round(seconds)))

    if seconds < 60:
        return f"{seconds}s"

    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes:02d}m"

    return f"{minutes}m {sec:02d}s"


def print_time_estimate(
    durations: List[float],
    completed: int,
    total: int,
) -> None:
    """
    Print the estimated remaining time and completion clock time for the
    images still to be analyzed, based on the average analysis time so far.

    The first image is excluded from the average: it includes the one-time
    OpenVINO graph compilation and would otherwise dominate the estimate.
    """
    samples = durations[1:]

    if len(samples) < TIMING_WARMUP_IMAGES:
        return

    remaining = total - completed

    if remaining <= 0:
        return

    avg_seconds = sum(samples) / len(samples)
    eta_seconds = avg_seconds * remaining
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)

    print(
        f"  Est. {format_duration(eta_seconds)} for {remaining} more image(s) "
        f"(~{avg_seconds:.1f}s/image, done ~{eta_time:%H:%M:%S})"
    )
