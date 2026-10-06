"""
Plain-text console output mechanics.

No colors and no animation — just the structural helpers that keep the
terminal output readable: section rules, aligned progress counters,
key-value summary lines, and consistent warning/error markers.
"""

from typing import Any

RULE_WIDTH = 72


def section(title: str) -> None:
    """Print a blank line followed by a section rule:

    -- Reading metadata ------------------------------------------
    """
    label = f"-- {title} "
    print("\n" + label + "-" * max(3, RULE_WIDTH - len(label)))


def counter(idx: int, total: int) -> str:
    """Right-aligned progress counter, e.g. '[  12/1234]'."""
    width = len(str(total)) if total > 0 else 1
    return f"[{idx:>{width}}/{total}]"


def fmt_int(value: int) -> str:
    """Thousands-separated integer, e.g. '1,234'."""
    return f"{value:,}"


def kv(key: str, value: Any) -> None:
    """Print one aligned 'key value' line, for summaries and status blocks."""
    print(f"  {key:<12}  {value}")


def note(message: str) -> None:
    print(f"  {message}")


def warn(message: str) -> None:
    print(f"  warning: {message}")


def error(message: str) -> None:
    print(f"  error: {message}")
