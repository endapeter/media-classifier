"""
Configuration and runtime environment bootstrap.

This module MUST be imported before any module that pulls in
transformers/optimum: it redirects the Hugging Face cache into ./models
and registers HEIC/HEIF support, so those effects are in place before the
first image is opened.

All tunable settings for the whole tool live here.
"""

import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
LOCAL_MODEL_DIR = MODEL_ROOT / "Qwen2.5-VL-7B-Instruct-int4-ov"

os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf_cache" / "hub"))


# ============================================================
# Optional HEIC/HEIF support
# ============================================================

HEIC_AVAILABLE = False
try:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        from pi_heif import register_heif_opener

    register_heif_opener()
    HEIC_AVAILABLE = True
except Exception as e:
    print(f"[Warning] HEIC/HEIF support disabled: {e}")
    print("         Install HEIC support with: python -m pip install pillow-heif")


# ============================================================
# Core configuration
# ============================================================

SOURCE_DIRECTORY = "./unorganized_images"
OUTPUT_DIRECTORY = "./organized_library"

MODEL_ID = (
    str(LOCAL_MODEL_DIR)
    if LOCAL_MODEL_DIR.exists()
    else "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov"
)

# Device priority: try each in order, fall back to the next if unavailable.
# Excluded devices and why:
# - NPU (Intel AI Boost): its compiler requires statically-shaped or
#   upper-bounded models; this export has fully unbounded dynamic dimensions,
#   so compilation always fails on NPU.
# - GPU.1 (discrete card, e.g. RTX 4050): this model needs ~13 GB of GPU
#   memory committed. A 6 GB discrete card silently spills the rest into
#   shared system RAM and pages it over PCIe, making inference pathologically
#   slow. The integrated GPU ("GPU") uses system memory directly with no VRAM
#   wall, so it is preferred; "CPU" is the last resort.
DEVICE_PRIORITY = ["GPU", "CPU"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

if HEIC_AVAILABLE:
    IMAGE_EXTENSIONS.update({".heic", ".heif"})

# If True, prints what would happen without moving/copying files.
DRY_RUN = False

# Time estimation: the first image is excluded from timing (it includes the
# one-time OpenVINO graph compilation and is far slower than every image
# after it); the next TIMING_WARMUP_IMAGES analyzed images form the initial
# average, and from then on each analysis prints an estimated remaining and
# completion time. The average keeps updating as more images are measured.
TIMING_WARMUP_IMAGES = 3

# Persist each image's VLM analysis to a checkpoint file as soon as it is
# produced. If the run is interrupted, the next run reuses the checkpointed
# analyses and only infers the remaining images, instead of starting over.
# Entries are keyed by source path and validated against file size and
# modification time, so a changed file is re-analyzed. The schema number
# also rejects entries written before the structured-slots refactor.
ANALYSIS_CHECKPOINT = True
CHECKPOINT_SCHEMA = 2

# If True, copies files instead of moving them.
# Safer for first runs, but can create duplicates if index is removed.
COPY_INSTEAD_OF_MOVE = False

# Skip hidden files/folders like .DS_Store, .git, .cache, etc.
SKIP_HIDDEN = True

# Use file modification time only if no EXIF/path/model year is available.
# Filesystem dates can be unreliable for copied files.
USE_FILESYSTEM_DATE_FALLBACK = True

# Fixed inference canvas. Qwen2.5-VL uses native dynamic resolution: the number
# of vision tokens (and therefore every downstream input shape) depends on the
# image's exact pixel dimensions. Each distinct shape forces OpenVINO to
# recompile the graph, which takes minutes for a 7B model. Padding every image
# onto one fixed canvas makes all shapes constant, so the graph compiles once
# and every image after the first is fast. Both dimensions are multiples of 28
# (Qwen2.5-VL patch size 14 x merge size 2) so the processor does not resize.
INFERENCE_CANVAS_WIDTH = 1008   # 28 * 36
INFERENCE_CANVAS_HEIGHT = 756   # 28 * 27

# Allow model to guess a year only if no EXIF/path year was found.
# Use with caution: VLMs can hallucinate dates.
ALLOW_MODEL_YEAR_FALLBACK = True

# Used when no year can be determined.
UNKNOWN_YEAR_FOLDER = "Unknown Year"

# Fallback folder for images whose date is only a year (no month), and for
# events whose name cannot be determined.
ASSORTED_FOLDER_NAME = "Assorted Photos"

# Generic camera/filename tokens that carry no content meaning.
GENERIC_NAME_WORDS = {"img", "image", "photo", "picture", "dsc", "dscn", "untitled"}

# Fixed classification vocabulary. The VLM must pick exactly one; free-form
# categories fragment semantically and were the main source of vague folders.
CATEGORY_VOCABULARY = [
    "people",
    "family",
    "friends",
    "baby",
    "celebration",
    "travel",
    "outdoors",
    "nature",
    "home",
    "food",
    "documents",
    "vehicle",
    "pets",
    "sports",
    "architecture",
    "other",
]

# ------------------------------------------------------------
# Event clustering (CLIP embeddings + agglomerative clustering)
# ------------------------------------------------------------

# Embedding model. Runs on CPU via transformers/torch; CLIP at 224px is only
# milliseconds per image. Cached under models/hf_cache via HF_HOME.
EMBED_MODEL_ID = "openai/clip-vit-base-patch32"
EMBED_INPUT_SIZE = 224
EMBED_BATCH_SIZE = 16

# Agglomerative clustering stops merging when cosine distance between
# clusters exceeds this. Photos are only ever clustered inside the same
# year-month bucket. Raise to merge more aggressively, lower to split.
CLUSTER_DISTANCE_THRESHOLD = 0.12

# Contact sheet (event naming) settings. 4 columns x 3 rows of 252px thumbs
# lands exactly on the 1008x756 inference canvas, so the Qwen graph never
# recompiles for cluster naming.
CONTACT_SHEET_MAX_FRAMES = 12
CONTACT_SHEET_THUMB_SIZE = 252

# ------------------------------------------------------------
# Filename and folder composition
# ------------------------------------------------------------

# Filename scheme: YYYYMMDD[_HHMMSS]_subject-setting[-action][_NNN].ext
# - Date/time prefix is ISO 8601 basic format and comes from EXIF when
#   available (time segment omitted when only a date is known).
# - Hyphens join words within a field; underscores join fields.
# - _NNN sequence numbers disambiguate identical stems (burst photos);
#   a short hash is only used if a sequence number still collides on disk.
MAX_EVENT_NAME_WORDS = 5
MAX_EVENT_NAME_LENGTH = 60
MAX_SLOT_WORDS = 4
MAX_DESCRIPTION_PARTS = 3  # subject + setting + action (+1 detail appended)
MAX_FOLDER_NAME_LENGTH = 80
MAX_FILENAME_LENGTH = 180

# Windows MAX_PATH guard: keep the total destination path well under 260.
MAX_TOTAL_PATH_LENGTH = 240

CURRENT_YEAR = datetime.now().year


# ============================================================
# Video configuration
# ============================================================

PROCESS_VIDEOS = True

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".webm",
    ".3gp",
    ".wmv",
    ".mts",
    ".m2ts",
}

# Videos go into: OUTPUT_DIRECTORY/Videos/Year/...
VIDEO_ROOT_FOLDER = "Videos"

# All videos are placed directly inside the year folder:
# source/trip/video.mp4 -> output/Videos/2023/video.mp4
VIDEO_PRESERVE_RELATIVE_FOLDERS = False

# Avoid duplicate year folders like Videos/2023/2023/video.mp4
VIDEO_REMOVE_YEAR_FOLDERS_FROM_PATH = True

# Use ffprobe if installed. If not installed, fallback date logic is used.
USE_FFPROBE_DATE = True


# ============================================================
# Shared regexes
# ============================================================

YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# Matches dates like:
# 2023-07-15, 2023_07_15, 2023.07.15, 20230715
FULL_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})(?:[-_./]?)(0?[1-9]|1[0-2])(?:[-_./]?)(0?[1-9]|[12]\d|3[01])(?!\d)"
)
