#!/usr/bin/env python3
"""
Image Library Organizer — entry point.

Organize images into:

Output/
  Year/
    MM Event Name/
      YYYYMMDD_HHMMSS_subject-setting_001.ext
      YYYYMMDD_HHMMSS_subject-setting_001.xmp   (XMP metadata sidecar)

Videos are organized in parallel into:

Output/
  Videos/
    Year/
      original_video_filename.ext

The tool is split into thematic modules:

    config.py     settings, environment bootstrap (HF cache, HEIC support)
    ui.py         aesthetic terminal layer; exports the styled `print`
    records.py    WhenInfo / ImageFacts / PlannedImage dataclasses
    helpers.py    word cleanup, year validation, stable hashing, ETA print
    facts.py      EXIF time/GPS/camera, filename/path dates, geocoding
    vlm.py        Qwen model loading, prompts, structured-slot inference
    clustering.py CLIP embeddings, event clustering, contact-sheet naming
    naming.py     filename/folder composition (pure code from slots)
    xmp.py        XMP sidecar writer (ISO 16684-1, Dublin Core, IPTC)
    state.py      move index, audit log, analysis checkpoint
    organizer.py   the image pipeline (phases 1-7)
    videos.py     video dating and moving (no AI)

High-level behavior:
- Recursively finds images; the year folder is the only date-based
  organization, derived from EXIF first, with filename/path/model/filesystem
  used only as year hints. When EXIF carries a full timestamp, the date AND
  time go into the filename so files sort chronologically.
- Deterministic metadata (EXIF time, GPS reverse-geocoded offline to
  city/country, camera make/model) is harvested before any inference.
- Images are embedded with a CLIP model and clustered (within each
  year-month) into events; one Qwen2.5-VL call per cluster, viewing a
  contact sheet, names the event. Each image then gets one Qwen2.5-VL call
  returning structured slots; filenames and folders are COMPOSED in code
  from those slots, never taken verbatim from the model.
- Every image gets a pure-Python XMP sidecar with Dublin Core and IPTC
  Extension metadata: title, description, keywords, location, provenance.
  Image bytes are never modified.
- Runs on the integrated Intel GPU if available, then CPU (the NPU and the
  discrete GPU are excluded; see config.py for why).
- Supports dry run, copy mode, logging, resume index, and analysis
  checkpointing.
"""

import config  # noqa: F401  (must run first: HF cache + HEIC bootstrap)
from config import OUTPUT_DIRECTORY, SOURCE_DIRECTORY
from organizer import organize_image_library
from ui import _ui_intro
from videos import process_videos


if __name__ == "__main__":
    try:
        _ui_intro()
    except KeyboardInterrupt:
        raise
    except Exception:
        pass

    organize_image_library(
        source_dir=SOURCE_DIRECTORY,
        target_dir=OUTPUT_DIRECTORY,
    )

    process_videos(
        source_dir=SOURCE_DIRECTORY,
        target_dir=OUTPUT_DIRECTORY,
    )
