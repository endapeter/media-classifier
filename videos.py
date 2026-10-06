"""
Video organizer: moves videos into OUTPUT_DIRECTORY/Videos/Year/...

This does not use the VLM.
"""

import os
import re
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Set

from config import (
    COPY_INSTEAD_OF_MOVE,
    DRY_RUN,
    PROCESS_VIDEOS,
    SKIP_HIDDEN,
    UNKNOWN_YEAR_FOLDER,
    USE_FFPROBE_DATE,
    USE_FILESYSTEM_DATE_FALLBACK,
    VIDEO_EXTENSIONS,
    VIDEO_PRESERVE_RELATIVE_FOLDERS,
    VIDEO_REMOVE_YEAR_FOLDERS_FROM_PATH,
    VIDEO_ROOT_FOLDER,
)
from facts import (
    extract_full_date_from_text,
    extract_year_from_text,
    get_filesystem_when,
)
from helpers import clean_existing_folder_part, valid_year
from naming import choose_destination
from records import WhenInfo
from state import append_log, load_index, save_index
from ui import counter, error, fmt_int, kv, section


# ============================================================
# Video date/year extraction
# ============================================================

def get_ffprobe_video_date(video_path: Path) -> Optional[date]:
    """
    Try to get video creation date using ffprobe.

    Requires ffmpeg/ffprobe installed.
    If unavailable, returns None and fallback logic is used.
    """
    if not shutil.which("ffprobe"):
        return None

    commands = [
        # Common location for video stream creation time.
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=creation_time",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        # Common location for container-level creation time.
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format_tags=creation_time",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="ignore",
            )

            value = result.stdout.strip()

            if not value:
                continue

            if value.startswith("0000"):
                continue

            # Example: 2023-07-15T12:34:56.000000Z
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"

            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                # Fallback for slightly non-standard timestamps.
                m = re.search(r"((?:19|20)\d{2})[-:](\d{2})[-:](\d{2})", value)
                if not m:
                    continue

                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))

            if valid_year(dt.year):
                return dt.date()

        except Exception:
            continue

    return None


def get_video_when(video_path: Path, source_path: Path) -> WhenInfo:
    """
    Determine year/date for a video.

    Priority:
        1. ffprobe metadata
        2. Full date in filename
        3. Full date in parent folders
        4. Year in filename
        5. Year in parent folders
        6. Filesystem date, if enabled
    """
    if USE_FFPROBE_DATE:
        ff_date = get_ffprobe_video_date(video_path)
        if ff_date:
            return WhenInfo(
                year=ff_date.year,
                date_token=ff_date.strftime("%Y%m%d"),
                source="ffprobe",
            )

    full_date = extract_full_date_from_text(video_path.name)
    if full_date:
        return WhenInfo(
            year=full_date.year,
            date_token=full_date.strftime("%Y%m%d"),
            source="filename",
        )

    try:
        rel_parent_parts = video_path.parent.relative_to(source_path).parts
    except Exception:
        rel_parent_parts = video_path.parent.parts

    for part in reversed(rel_parent_parts):
        full_date = extract_full_date_from_text(part)
        if full_date:
            return WhenInfo(
                year=full_date.year,
                date_token=full_date.strftime("%Y%m%d"),
                source="path",
            )

    year = extract_year_from_text(video_path.name)
    if year:
        return WhenInfo(
            year=year,
            date_token=f"{year:04d}",
            source="filename_year",
        )

    for part in reversed(rel_parent_parts):
        year = extract_year_from_text(part)
        if year:
            return WhenInfo(
                year=year,
                date_token=f"{year:04d}",
                source="path_year",
            )

    if USE_FILESYSTEM_DATE_FALLBACK:
        fs_when = get_filesystem_when(video_path)
        if fs_when.year is not None:
            return fs_when

    return WhenInfo(year=None, date_token="undated", source="none")


# ============================================================
# Video organizer
# ============================================================

def process_videos(source_dir: str, target_dir: str) -> None:
    """
    Move videos into:

        OUTPUT_DIRECTORY/VIDEO_ROOT_FOLDER/Year/...

    This does not use the VLM.
    """
    if not PROCESS_VIDEOS:
        return

    source_path = Path(source_dir).expanduser().resolve()
    target_path = Path(target_dir).expanduser().resolve()

    if not source_path.exists():
        return

    if source_path == target_path:
        raise ValueError("Source and output directory must be different.")

    try:
        target_is_inside_source = target_path.is_relative_to(source_path)
    except AttributeError:
        target_is_inside_source = str(target_path).startswith(str(source_path) + os.sep)

    video_files: List[Path] = []

    for p in sorted(source_path.rglob("*")):
        try:
            if not p.is_file():
                continue

            if p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            # Avoid processing already organized output files.
            if target_is_inside_source:
                if p == target_path or target_path in p.parents:
                    continue

            if SKIP_HIDDEN:
                try:
                    rel_parts = p.relative_to(source_path).parts
                except ValueError:
                    rel_parts = p.parts

                if any(part.startswith(".") for part in rel_parts):
                    continue

            video_files.append(p)

        except OSError:
            continue

    section("Videos")
    print(f"Found {fmt_int(len(video_files))} video(s) in {source_path}")

    if not video_files:
        return

    index = {} if DRY_RUN else load_index(target_path)

    planned: Set[Path] = set()

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, video_path in enumerate(video_files, start=1):
        try:
            try:
                rel = video_path.relative_to(source_path)
            except ValueError:
                rel = video_path

            tag = counter(idx, len(video_files))
            source_key = str(video_path)

            if not DRY_RUN and source_key in index:
                existing_destination = Path(index[source_key])

                if existing_destination.exists():
                    print(f"{tag} {rel}  (already processed)")
                    skipped_count += 1
                    continue

            try:
                when = get_video_when(video_path, source_path)

                year_folder = str(when.year) if when.year is not None else UNKNOWN_YEAR_FOLDER
                dest_root = target_path / VIDEO_ROOT_FOLDER / year_folder

                if VIDEO_PRESERVE_RELATIVE_FOLDERS:
                    try:
                        rel_dir = video_path.parent.relative_to(source_path)
                    except ValueError:
                        rel_dir = Path("")

                    parts = list(rel_dir.parts)

                    if VIDEO_REMOVE_YEAR_FOLDERS_FROM_PATH and when.year is not None:
                        year_str = str(when.year)
                        parts = [p for p in parts if p != year_str]

                    clean_parts = [
                        clean_existing_folder_part(p)
                        for p in parts
                        if p not in ("", ".", "..")
                    ]

                    dest_folder = dest_root / Path(*clean_parts) if clean_parts else dest_root
                else:
                    dest_folder = dest_root

                # Keep original video filename, but avoid collisions.
                dest_path = choose_destination(
                    dest_folder=dest_folder,
                    filename=video_path.name,
                    planned=planned,
                )

                try:
                    shown_destination = dest_path.relative_to(target_path)
                except ValueError:
                    shown_destination = dest_path

                print(f"{tag} {rel} -> {shown_destination}")

                if not DRY_RUN:
                    dest_folder.mkdir(parents=True, exist_ok=True)

                    if COPY_INSTEAD_OF_MOVE:
                        shutil.copy2(video_path, dest_path)
                        action = "copied"
                    else:
                        shutil.move(str(video_path), str(dest_path))
                        action = "moved"

                    index[source_key] = str(dest_path)
                    save_index(target_path, index)

                    append_log(
                        target_path,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "source": str(video_path),
                            "destination": str(dest_path),
                            "action": action,
                            "type": "video",
                            "when": when.__dict__,
                        },
                    )

                planned.add(dest_path)
                success_count += 1

            except Exception as e:
                failed_count += 1
                error(f"failed to process video {video_path.name}: {e}")

                if not DRY_RUN:
                    append_log(
                        target_path,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "source": str(video_path),
                            "type": "video",
                            "error": str(e),
                        },
                    )

        except Exception as e:
            failed_count += 1

            try:
                shown = video_path.relative_to(source_path)
            except ValueError:
                shown = video_path

            error(f"unexpected failure while processing video {shown}: {e}")

            if not DRY_RUN:
                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(video_path),
                        "type": "video",
                        "error": str(e),
                        "phase": "video_processing",
                    },
                )

    section("Video summary")

    kv("Processed", fmt_int(success_count))
    kv("Skipped", fmt_int(skipped_count))
    kv("Failed", fmt_int(failed_count))

    if DRY_RUN:
        print("  dry run - no videos were moved or copied")
