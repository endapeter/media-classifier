"""
Filename and folder composition.

Filenames and folder names are always COMPOSED here in code from the
structured VLM slots and deterministic facts — never taken verbatim from
model output, and never from the original filename.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import (
    ASSORTED_FOLDER_NAME,
    CATEGORY_VOCABULARY,
    MAX_DESCRIPTION_PARTS,
    MAX_EVENT_NAME_LENGTH,
    MAX_EVENT_NAME_WORDS,
    MAX_FILENAME_LENGTH,
    MAX_FOLDER_NAME_LENGTH,
    MAX_SLOT_WORDS,
)
from helpers import (
    remove_generic_words,
    remove_year_words,
    stable_hash,
    words_from_text,
)
from records import PlannedImage


def sanitize_slot_words(value: Any, max_words: int = MAX_SLOT_WORDS) -> List[str]:
    """Clean one slot value into a short list of safe lowercase words."""
    words = words_from_text(value)
    words = remove_year_words(words)
    words = remove_generic_words(words)

    # Remove pure numeric words and single letters (e.g. the "s" left
    # behind by "New Year's" after punctuation is stripped).
    words = [w for w in words if not re.fullmatch(r"\d+", w) and len(w) > 1]

    return words[:max_words]


def sanitize_event_name(raw: Any) -> str:
    """
    Convert model output into a safe event folder name part.

    Returns "" when the output is unusable; the caller then falls back to
    the member images' category or the assorted folder.
    """
    words = sanitize_slot_words(raw, MAX_EVENT_NAME_WORDS)

    if len(words) < 2:
        return ""

    name = " ".join(w.capitalize() if w.islower() else w for w in words)

    # Final filesystem safety.
    name = re.sub(r'[<>:"/\\|?*]+', " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    if len(name) > MAX_EVENT_NAME_LENGTH:
        name = name[:MAX_EVENT_NAME_LENGTH].rstrip()

    return name


def compose_description_parts(record: PlannedImage) -> List[str]:
    """
    Hyphen-joined description fields for the filename.

    Words already used by the event name are removed, so the filename never
    repeats the folder it lives in (the old scheme produced names like
    "newborn_baby_sleeping_newborn_baby_290b84.jpg").
    """
    event_words = set(words_from_text(record.event_name))

    parts: List[str] = []

    for key in ("subject", "setting", "action"):
        words = [w for w in sanitize_slot_words(record.slots.get(key)) if w not in event_words]

        if words:
            parts.append("-".join(words))

    # One distinctive detail, when there is room for it.
    if len(parts) < MAX_DESCRIPTION_PARTS:
        details = record.slots.get("details")

        if isinstance(details, (list, tuple)):
            for detail in details:
                words = [
                    w
                    for w in sanitize_slot_words(detail, max_words=2)
                    if w not in event_words
                ]

                if words:
                    parts.append("-".join(words))
                    break

    return parts[:MAX_DESCRIPTION_PARTS]


def month_token(record: PlannedImage) -> Optional[str]:
    """Two-digit month when the date is at least day-precision, else None."""
    dt = record.facts.dt

    if dt is not None:
        return dt.strftime("%m")

    token = record.facts.when.date_token

    if re.fullmatch(r"\d{8}", token):
        return token[4:6]

    return None


def compose_event_folder(record: PlannedImage) -> str:
    """
    Folder below the year: "MM Event Name" when both month and event are
    known; otherwise the assorted folder for that year.
    """
    month = month_token(record)

    if month is None:
        return ASSORTED_FOLDER_NAME

    return f"{month} {record.event_name}"[:MAX_FOLDER_NAME_LENGTH].rstrip()


def compose_filename(record: PlannedImage) -> str:
    """
    Build the final filename:

        YYYYMMDD_HHMMSS_subject-setting[-action][_NNN].ext

    - The date/time prefix is ISO 8601 basic format and comes from EXIF;
      the time segment is omitted when only a date is known (mixed prefixes
      still sort correctly by date).
    - Hyphens join words within a field; underscores join fields.
    - Sequence numbers (_001) are added by the caller for identical stems;
      no hash is baked in by default.
    """
    ext = record.img_path.suffix.lower() or ".jpg"
    facts = record.facts

    prefix = None

    if facts.dt is not None:
        prefix = facts.dt.strftime("%Y%m%d_%H%M%S")
    else:
        token = facts.when.date_token

        if re.fullmatch(r"\d{8}", token):
            prefix = token
        elif re.fullmatch(r"\d{4}", token):
            prefix = token

    desc_parts = compose_description_parts(record)

    if not desc_parts:
        # Fall back to the category, then to a generic word.
        category = str(record.slots.get("category") or "").strip().lower()
        desc_parts = [category if category in CATEGORY_VOCABULARY else "photo"]

    base = "_".join([p for p in (prefix, "_".join(desc_parts)) if p])

    # Leave room for extension, a sequence number, and a rare hash suffix.
    max_base = MAX_FILENAME_LENGTH - len(ext) - 12

    if len(base) > max_base:
        base = base[:max_base].rstrip("_-")

    return f"{base}{ext}"


def assign_sequence_numbers(
    moves: List[Dict[str, Any]],
) -> None:
    """
    Add _NNN sequence suffixes when several images would land in the same
    folder with the same filename stem (burst photos).

    `moves` items are dicts with "dest_folder" (str), "filename" (str), and a
    "record" (PlannedImage); the "filename" is updated in place.
    """
    stems: Dict[Tuple[str, str], int] = {}

    for move in moves:
        key = (str(move["dest_folder"]), move["filename"])
        stems[key] = stems.get(key, 0) + 1

    seen: Dict[Tuple[str, str], int] = {}

    for move in moves:
        key = (str(move["dest_folder"]), move["filename"])

        if stems[key] < 2:
            continue

        seen[key] = seen.get(key, 0) + 1

        p = Path(move["filename"])
        move["filename"] = f"{p.stem}_{seen[key]:03d}{p.suffix}"


def choose_destination(
    dest_folder: Path,
    filename: str,
    planned: Set[Path],
    img_path: Optional[Path] = None,
) -> Path:
    """
    Avoid filename collisions with existing files and files planned
    earlier in the same run.

    Same-run duplicates are already disambiguated with _NNN sequence
    numbers before this runs, so a collision here means a pre-existing file
    on disk: append the source's stable hash, then counters as a last resort.
    """
    candidate = dest_folder / filename

    if not candidate.exists() and candidate not in planned:
        return candidate

    stem = Path(filename).stem
    ext = Path(filename).suffix

    if img_path is not None:
        hashed = dest_folder / f"{stem}_{stable_hash(img_path)}{ext}"

        if not hashed.exists() and hashed not in planned:
            return hashed

    counter = 1

    while True:
        candidate = dest_folder / f"{stem}_{counter:02d}{ext}"

        if not candidate.exists() and candidate not in planned:
            return candidate

        counter += 1
