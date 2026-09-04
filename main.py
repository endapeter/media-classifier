#!/usr/bin/env python3
"""
Organize images into:

    Output/
        Year/
            Thematic Folder/
                content_described_filename.ext

Videos are organized in parallel into:

    Output/
        Videos/
            Year/
                original_video_filename.ext

Features:
- Recursively finds images.
- The year folder is the only date-based organization; it is derived from EXIF
  first, with filename/path/model/filesystem used only as year hints.
- Uses Qwen2.5-VL via OpenVINO to analyze the image content itself and propose
  both the thematic folder and a content-based filename. Original filenames are
  never used for naming or theming.
- A thematic folder is only created when at least MIN_IMAGES_PER_THEME_FOLDER
  images share it (within a year); sparser themes are merged into a per-year
  "Assorted Photos" folder so no folder holds just one or two images.
- Runs on the integrated Intel GPU if available, then CPU. The NPU is not
  used (its compiler rejects this model's fully dynamic input shapes), and the
  discrete GPU is not used (the model needs ~13 GB of GPU memory, which would
  page over PCIe on a 6 GB card).
- Enforces safer folder and filename naming.
- Supports dry run, copy mode, logging, and resume index.
- Moves videos into a parallel Videos/Year structure without AI classification.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
LOCAL_MODEL_DIR = MODEL_ROOT / "Qwen2.5-VL-7B-Instruct-int4-ov"

os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf_cache" / "hub"))

import json
import re
import shutil
import hashlib
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any, Set, List

from PIL import Image
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor


# ============================================================
# Configuration
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

# If True, prints what would happen without moving/copying files.
DRY_RUN = False

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
INFERENCE_CANVAS_WIDTH = 1008  # 28 * 36
INFERENCE_CANVAS_HEIGHT = 756  # 28 * 27

# Allow model to guess a year only if no EXIF/path year was found.
# Use with caution: VLMs can hallucinate dates.
ALLOW_MODEL_YEAR_FALLBACK = True

# Folder style:
# "title_spaces"       -> "Family Beach Vacation"
# "lower_underscores"  -> "family_beach_vacation"
FOLDER_STYLE = "title_spaces"

# Used when no year can be determined.
UNKNOWN_YEAR_FOLDER = "Unknown Year"

# Fallback theme if model output is unusable.
DEFAULT_THEME_WORDS = ["general", "photo", "collection"]

# Generic camera/filename tokens that carry no content meaning.
GENERIC_NAME_WORDS = {"img", "image", "photo", "picture", "dsc", "dscn", "untitled"}

# Thematic folder rules.
MIN_THEME_WORDS = 2
MAX_THEME_WORDS = 6

# A thematic folder is only created when at least this many images share it
# (within the same year). Smaller groups are merged into ASSORTED_FOLDER_NAME
# for that year, so the library never ends up with folders holding just one
# or two images. Exception: the assorted folder itself may be small if an
# entire year has fewer than this many images.
MIN_IMAGES_PER_THEME_FOLDER = 10
ASSORTED_FOLDER_NAME = "Assorted Photos"

# Filename rules.
MAX_THEME_SLUG_WORDS_IN_FILENAME = 2
MAX_DESCRIPTION_WORDS = 8
MAX_FOLDER_NAME_LENGTH = 80
MAX_FILENAME_LENGTH = 180

CURRENT_YEAR = datetime.now().year

YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# Matches dates like:
# 2023-07-15, 2023_07_15, 2023.07.15, 20230715
FULL_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})(?:[-_./]?)(0?[1-9]|1[0-2])(?:[-_./]?)(0?[1-9]|[12]\d|3[01])(?!\d)"
)


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
# Data structures
# ============================================================

@dataclass
class WhenInfo:
    year: Optional[int]
    date_token: str
    source: str


@dataclass
class PlannedImage:
    """An analyzed image awaiting its final folder assignment and move."""
    img_path: Path
    when: WhenInfo
    theme_raw: str
    filename_raw: str
    prediction: Dict[str, Any]
    raw_response: str


# ============================================================
# Basic helpers
# ============================================================

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
# Date/year extraction
# ============================================================

def parse_exif_datetime(value: Any) -> Optional[date]:
    """Parse common EXIF date formats."""
    if not isinstance(value, str):
        return None

    value = value.strip()

    for fmt in (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    # Fallback regex for embedded YYYY-MM-DD / YYYY:MM:DD.
    m = re.search(r"((?:19|20)\d{2})[:\-/](\d{2})[:\-/](\d{2})", value)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    return None


def get_exif_date(image: Image.Image) -> Optional[date]:
    """
    Extract date from EXIF.
    Priority:
        DateTimeOriginal
        DateTimeDigitized
        DateTime
    """
    try:
        exif = image.getexif()
        if not exif:
            return None

        values = []

        # Base IFD tags.
        for tag in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
            try:
                if tag in exif:
                    values.append(exif[tag])
            except Exception:
                pass

        # Exif IFD tags.
        try:
            exif_ifd = exif.get_ifd(0x8769)
            for tag in (36867, 36868):
                if tag in exif_ifd:
                    values.append(exif_ifd[tag])
        except Exception:
            pass

        for value in values:
            parsed = parse_exif_datetime(value)
            if parsed and valid_year(parsed.year):
                return parsed

    except Exception:
        return None

    return None


def extract_full_date_from_text(text: str) -> Optional[date]:
    """Extract a full date from text if present."""
    for m in FULL_DATE_RE.finditer(text):
        try:
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))
            dt = date(y, mo, d)
            if valid_year(y):
                return dt
        except ValueError:
            continue
    return None


def extract_year_from_text(text: str) -> Optional[int]:
    """Extract a plausible year from text."""
    for m in YEAR_RE.finditer(text):
        y = int(m.group(1))
        if valid_year(y):
            return y
    return None


def extract_deterministic_when(
    img_path: Path,
    image: Image.Image,
    source_path: Path
) -> WhenInfo:
    """
    Determine date/year without using the VLM.

    Priority:
        1. EXIF date
        2. Full date in filename
        3. Full date in parent folders inside source directory
        4. Year in filename
        5. Year in parent folders inside source directory
    """

    # 1. EXIF.
    exif_date = get_exif_date(image)
    if exif_date:
        return WhenInfo(
            year=exif_date.year,
            date_token=exif_date.strftime("%Y%m%d"),
            source="exif",
        )

    # 2. Full date in filename.
    full_date = extract_full_date_from_text(img_path.name)
    if full_date:
        return WhenInfo(
            year=full_date.year,
            date_token=full_date.strftime("%Y%m%d"),
            source="filename",
        )

    # Restrict path-based search to folders inside the source directory.
    try:
        rel_parent_parts = img_path.parent.relative_to(source_path).parts
    except Exception:
        rel_parent_parts = img_path.parent.parts

    # 3. Full date in parent folder names.
    for part in reversed(rel_parent_parts):
        full_date = extract_full_date_from_text(part)
        if full_date:
            return WhenInfo(
                year=full_date.year,
                date_token=full_date.strftime("%Y%m%d"),
                source="path",
            )

    # 4. Year in filename.
    year = extract_year_from_text(img_path.name)
    if year:
        return WhenInfo(
            year=year,
            date_token=f"{year:04d}",
            source="filename_year",
        )

    # 5. Year in parent folder names.
    for part in reversed(rel_parent_parts):
        year = extract_year_from_text(part)
        if year:
            return WhenInfo(
                year=year,
                date_token=f"{year:04d}",
                source="path_year",
            )

    return WhenInfo(year=None, date_token="undated", source="none")


def get_filesystem_when(path: Path) -> WhenInfo:
    """Fallback using file modification time."""
    try:
        st = path.stat()
        dt = datetime.fromtimestamp(st.st_mtime)
        if valid_year(dt.year):
            return WhenInfo(
                year=dt.year,
                date_token=dt.strftime("%Y%m%d"),
                source="filesystem",
            )
    except Exception:
        pass

    return WhenInfo(year=None, date_token="undated", source="none")


def finalize_when(
    deterministic_when: WhenInfo,
    observed_year: Optional[int],
    img_path: Path
) -> WhenInfo:
    """
    Final year/date decision.

    Priority:
        1. EXIF/path-derived deterministic date/year
        2. Model-observed year, if allowed
        3. Filesystem date, if allowed
        4. Unknown
    """
    if deterministic_when.year is not None:
        return deterministic_when

    if ALLOW_MODEL_YEAR_FALLBACK and observed_year is not None and valid_year(observed_year):
        return WhenInfo(
            year=observed_year,
            date_token=f"{observed_year:04d}",
            source="model",
        )

    if USE_FILESYSTEM_DATE_FALLBACK:
        fs_when = get_filesystem_when(img_path)
        if fs_when.year is not None:
            return fs_when

    return WhenInfo(year=None, date_token="undated", source="none")


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
# VLM prompting and parsing
# ============================================================

def prepare_for_inference(image: Image.Image) -> Image.Image:
    """
    Downscale and letterbox onto the fixed inference canvas.

    The image is scaled down to fit inside the canvas (aspect ratio
    preserved), snapped to a multiple of 28 pixels per side, then centered
    on a black canvas of exactly INFERENCE_CANVAS_WIDTH x
    INFERENCE_CANVAS_HEIGHT. Constant input dimensions mean a constant
    vision-token count, so the OpenVINO graph compiles once instead of
    once per image.
    """
    canvas_w = INFERENCE_CANVAS_WIDTH
    canvas_h = INFERENCE_CANVAS_HEIGHT

    scale = min(canvas_w / image.width, canvas_h / image.height, 1.0)
    new_w = max(28, (round(image.width * scale) // 28) * 28)
    new_h = max(28, (round(image.height * scale) // 28) * 28)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    canvas.paste(resized, ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2))
    return canvas


def get_vlm_prompt() -> str:
    return """Analyze this image and name it based only on what is actually visible in it.
Return ONLY one valid JSON object. Do not use Markdown. Do not add comments.

{
  "theme_folder": "several word descriptive theme",
  "filename": "short content based filename",
  "observed_year": null
}

Rules:
1. "theme_folder" must be 2-6 words, broad but specific, no year, no slashes.
   Describe the subject/setting of the image, e.g.:
     "family beach vacation"
     "home renovation project"
     "scanned financial documents"
     "winter mountain hiking"
2. "filename" must be 3-8 lowercase words describing what is actually shown
   in this specific image (subject, setting, action). No dates, no numbers
   from timestamps, no file extension, no slashes.
   Good examples:
     "sunset over rocky pier"
     "birthday cake with candles"
     "group hiking snowy mountain"
3. "observed_year" must be an integer only if a year is clearly visible in the image.
   Otherwise return null.
"""


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from model output.
    Handles:
      - Markdown fences
      - Extra surrounding text
      - Trailing commas
    """
    if not response:
        return None

    text = response.strip()

    # Remove common Markdown code fences.
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("`", "")

    # Direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try first { to last }.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]

        try:
            return json.loads(snippet)
        except Exception:
            pass

        # Remove trailing commas before } or ].
        repaired = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            return json.loads(repaired)
        except Exception:
            pass

    # Last resort: regex object extraction.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        snippet = m.group(0)

        try:
            return json.loads(snippet)
        except Exception:
            pass

        repaired = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            return json.loads(repaired)
        except Exception:
            return None

    return None


def predict_metadata(
    model: OVModelForVisualCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    max_new_tokens: int = 96,
) -> Tuple[Dict[str, Any], str]:
    """
    Run VLM inference and return parsed JSON plus raw response.
    """
    prompt = get_vlm_prompt()

    # Prefer Qwen-style chat template.
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for older processor versions.
        text = prompt

    inputs = processor(images=image, text=text, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        # The expected JSON response is ~50-70 tokens; generation is
        # autoregressive, so every extra token adds a decode step.
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    # Important: decode only newly generated tokens.
    # Otherwise the prompt's example JSON may be mistakenly parsed.
    try:
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
    except Exception:
        generated_ids = output_ids

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    data = parse_json_response(response) or {}
    return data, response


def extract_observed_year(data: Dict[str, Any]) -> Optional[int]:
    """Extract model-observed year if valid."""
    for key in ("observed_year", "year", "image_year"):
        if key not in data:
            continue

        raw = data[key]
        if raw is None:
            continue

        s = str(raw).strip()
        if s.lower() in {"", "none", "null"}:
            continue

        try:
            y = int(float(s))
        except Exception:
            m = YEAR_RE.search(s)
            if not m:
                continue
            y = int(m.group(1))

        if valid_year(y):
            return y

    return None


# ============================================================
# Folder and filename normalization
# ============================================================

def normalize_theme_folder(raw: Any) -> Tuple[str, List[str]]:
    """
    Convert model output into a safe thematic folder name.

    The folder reflects image content only; if the model output is
    unusable, a generic theme is used rather than the original filename.

    Returns:
        folder_name, theme_words
    """
    words = words_from_text(raw)
    words = remove_year_words(words)
    words = remove_generic_words(words)

    # Remove pure numeric words from theme folders.
    words = [w for w in words if not re.fullmatch(r"\d+", w)]

    if not words:
        words = list(DEFAULT_THEME_WORDS)

    # Ensure the theme folder has several words if required.
    if len(words) < MIN_THEME_WORDS:
        extras = ["photos", "collection", "archive"]
        for extra in extras:
            if len(words) >= MIN_THEME_WORDS:
                break
            if extra not in words:
                words.append(extra)

    words = words[:MAX_THEME_WORDS]

    if FOLDER_STYLE == "lower_underscores":
        name = "_".join(words)
    else:
        # Simple human-readable title case.
        name = " ".join(w.capitalize() if w.islower() else w for w in words)

    # Final filesystem safety.
    name = re.sub(r'[<>:"/\\|?*]+', " ", name)

    if FOLDER_STYLE == "lower_underscores":
        name = re.sub(r"\s+", "_", name)
    else:
        name = re.sub(r"\s+", " ", name)

    name = name.strip()

    if len(name) > MAX_FOLDER_NAME_LENGTH:
        name = name[:MAX_FOLDER_NAME_LENGTH].rstrip()

    if not name:
        words = list(DEFAULT_THEME_WORDS)
        if FOLDER_STYLE == "lower_underscores":
            name = "_".join(words)
        else:
            name = " ".join(w.capitalize() for w in words)

    return name, words


def assign_theme_folders(
    records: List[PlannedImage],
) -> List[Tuple[PlannedImage, str, List[str]]]:
    """
    Decide the final thematic folder for each analyzed image.

    Returns (record, folder_name, theme_words) triples. A theme that would
    hold fewer than MIN_IMAGES_PER_THEME_FOLDER images within its year is
    merged into ASSORTED_FOLDER_NAME instead, so no thematic folder ends up
    with just one or two images.

    Note: theme_words always describe the image's own theme; they are kept
    as-is for merged images so filenames still reflect the content.
    """
    folder_counts: Dict[Tuple[Optional[int], str], int] = {}
    resolved: List[Tuple[PlannedImage, str, List[str]]] = []

    for record in records:
        theme_folder, theme_words = normalize_theme_folder(record.theme_raw)
        resolved.append((record, theme_folder, theme_words))
        key = (record.when.year, theme_folder)
        folder_counts[key] = folder_counts.get(key, 0) + 1

    merged_count = 0
    final: List[Tuple[PlannedImage, str, List[str]]] = []
    for record, theme_folder, theme_words in resolved:
        if folder_counts[(record.when.year, theme_folder)] >= MIN_IMAGES_PER_THEME_FOLDER:
            final.append((record, theme_folder, theme_words))
        else:
            merged_count += 1
            final.append((record, ASSORTED_FOLDER_NAME, theme_words))

    if merged_count:
        merged_names = sorted(
            folder
            for (_, folder), count in folder_counts.items()
            if count < MIN_IMAGES_PER_THEME_FOLDER
        )
        print(
            f"\nMerging {merged_count} images from {len(merged_names)} folders "
            f"with fewer than {MIN_IMAGES_PER_THEME_FOLDER} images into "
            f"'{ASSORTED_FOLDER_NAME}':"
        )
        for name in merged_names:
            print(f"  - {name}")
        print()

    return final


def make_slug(
    text: Any,
    max_words: int = MAX_DESCRIPTION_WORDS
) -> str:
    """
    Create a lowercase underscore slug for filenames.
    Falls back to a generic word if the model output is unusable —
    never to the original filename.
    """
    words = remove_year_words(words_from_text(text))
    words = remove_generic_words(words)

    if not words:
        words = ["photo"]

    words = words[:max_words]
    slug = "_".join(words)

    # Keep filename slugs reasonably short.
    if len(slug) > 80:
        slug = slug[:80].rstrip("_")

    return slug


def build_filename(
    img_path: Path,
    theme_words: List[str],
    filename_raw: Any
) -> str:
    """
    Build a controlled, content-based filename.

    Default scheme:
        {theme_slug}_{description_slug}_{hash6}.ext

    The year lives only in the folder structure; the filename itself
    describes the image content. A short hash keeps names unique.

    Examples:
        family_beach_sunset_over_rocky_pier_a1b2c3.jpg
        home_renovation_new_kitchen_flooring_9f3e2a.jpg
    """
    ext = img_path.suffix.lower() or ".jpg"

    theme_slug = "_".join(theme_words[:MAX_THEME_SLUG_WORDS_IN_FILENAME]) if theme_words else ""
    desc_slug = make_slug(filename_raw)

    base_parts = [theme_slug, desc_slug]

    base = "_".join([p for p in base_parts if p])
    base = re.sub(r"_+", "_", base).strip("_")

    h = stable_hash(img_path)

    # Leave room for hash, extension, and possible collision counter.
    max_base = MAX_FILENAME_LENGTH - len(ext) - len(h) - 5
    if max_base < 20:
        max_base = 20

    if len(base) > max_base:
        base = base[:max_base].rstrip("_")

    return f"{base}_{h}{ext}"


def choose_destination(
    dest_folder: Path,
    filename: str,
    planned: Set[Path]
) -> Path:
    """
    Avoid filename collisions with existing files and files planned
    earlier in the same run.
    """
    candidate = dest_folder / filename

    if not candidate.exists() and candidate not in planned:
        return candidate

    stem = Path(filename).stem
    ext = Path(filename).suffix
    counter = 1

    while True:
        candidate = dest_folder / f"{stem}_{counter:02d}{ext}"
        if not candidate.exists() and candidate not in planned:
            return candidate
        counter += 1


# ============================================================
# Logging and resume index
# ============================================================

def load_index(target_path: Path) -> Dict[str, str]:
    index_path = target_path / "_organize_index.json"

    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    return {}


def save_index(target_path: Path, index: Dict[str, str]) -> None:
    index_path = target_path / "_organize_index.json"

    try:
        target_path.mkdir(parents=True, exist_ok=True)
        tmp_path = index_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(index_path)
    except Exception as e:
        print(f"  [Warning] Could not save index: {e}")


def append_log(target_path: Path, entry: Dict[str, Any]) -> None:
    log_path = target_path / "_organize_log.jsonl"

    try:
        target_path.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [Warning] Could not write log: {e}")


# ============================================================
# Model loading
# ============================================================

def load_model() -> Tuple[AutoProcessor, OVModelForVisualCausalLM]:
    print(f"Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = None
    for device in DEVICE_PRIORITY:
        # Intel GPUs cap single allocations; large vision-encoder buffers
        # can exceed that cap unless large allocations are enabled. The
        # option only exists in newer OpenVINO builds, so try it first and
        # fall back to a plain load if the plugin rejects it.
        attempts = [{}]
        if device.startswith("GPU"):
            attempts.insert(0, {"ov::intel_gpu::hint::enable_large_allocations": True})

        for ov_config in attempts:
            print(f"Loading OpenVINO model on {device}...")
            try:
                model = OVModelForVisualCausalLM.from_pretrained(
                    MODEL_ID,
                    device=device,
                    ov_config=ov_config,
                )
                print(f"Model loaded on {device}.")
                break
            except Exception as e:
                print(f"Failed to load on {device}: {e}")
                if ov_config:
                    print("Retrying without large-allocation hint...")
                else:
                    print("Trying next device...")

        if model is not None:
            break

    if model is None:
        raise RuntimeError(
            f"Could not load the model on any device: {DEVICE_PRIORITY}"
        )

    return processor, model


# ============================================================
# Main image organizer
# ============================================================

def organize_image_library(source_dir: str, target_dir: str) -> None:
    source_path = Path(source_dir).expanduser().resolve()
    target_path = Path(target_dir).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    if source_path == target_path:
        raise ValueError("Source and output directory must be different.")

    # Only skip files inside target if target is inside source.
    # If target is a parent of source, we still want to process source files.
    try:
        target_is_inside_source = target_path.is_relative_to(source_path)
    except AttributeError:
        target_is_inside_source = str(target_path).startswith(str(source_path) + os.sep)

    if target_is_inside_source:
        print("Notice: output directory is inside source directory. Output files will be skipped during scanning.")

    # ------------------------------------------------------------
    # Find images recursively.
    # ------------------------------------------------------------
    image_files: List[Path] = []

    for p in sorted(source_path.rglob("*")):
        try:
            if not p.is_file():
                continue

            if p.suffix.lower() not in IMAGE_EXTENSIONS:
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

            image_files.append(p)

        except OSError:
            continue

    print(f"Found {len(image_files)} images in {source_path}.\n")

    if not image_files:
        return

    processor, model = load_model()

    index = {} if DRY_RUN else load_index(target_path)
    planned: Set[Path] = set()

    success_count = 0
    skipped_count = 0
    failed_count = 0

    # ------------------------------------------------------------
    # Phase 1: analyze every image without moving anything yet.
    # Folder assignment needs the complete picture, because a thematic
    # folder is only kept when enough images share it.
    # ------------------------------------------------------------
    analyzed: List[PlannedImage] = []

    for idx, img_path in enumerate(image_files, start=1):
        try:
            rel = img_path.relative_to(source_path)
        except ValueError:
            rel = img_path

        print(f"[{idx}/{len(image_files)}] Analyzing: {rel}")

        source_key = str(img_path)

        # Resume support.
        if not DRY_RUN and source_key in index:
            existing_destination = Path(index[source_key])
            if existing_destination.exists():
                print(f"  -> Already processed: {existing_destination}")
                skipped_count += 1
                continue

        try:
            # ----------------------------------------------------
            # Open image and extract deterministic date/year.
            # ----------------------------------------------------
            with Image.open(img_path) as img:
                deterministic_when = extract_deterministic_when(
                    img_path,
                    img,
                    source_path,
                )
                rgb_image = img.convert("RGB")

            # Letterbox onto the fixed inference canvas so every image
            # produces identical input shapes; otherwise OpenVINO
            # recompiles the graph for each new image geometry.
            inference_image = prepare_for_inference(rgb_image)

            # ----------------------------------------------------
            # VLM inference.
            # ----------------------------------------------------
            try:
                prediction, raw_response = predict_metadata(
                    model,
                    processor,
                    inference_image,
                )
            except Exception as e:
                print(
                    f"  [Warning] Model inference failed: {e}\n"
                    f"  Falling back to a generic theme/filename for this image."
                )
                prediction, raw_response = {}, ""

            observed_year = extract_observed_year(prediction)

            when = finalize_when(
                deterministic_when=deterministic_when,
                observed_year=observed_year,
                img_path=img_path,
            )

            # ----------------------------------------------------
            # Normalize model output. Naming comes from image content
            # only — never from the original filename.
            # ----------------------------------------------------
            theme_raw = (
                prediction.get("theme_folder")
                or prediction.get("category_path")
                or ""
            )

            # If the model accidentally returns an extension or path,
            # use the stem of what it returned.
            filename_raw = prediction.get("filename") or prediction.get("description")
            if filename_raw:
                filename_raw = Path(str(filename_raw)).stem

            print(f"  Date token: {when.date_token} (source: {when.source})")

            analyzed.append(
                PlannedImage(
                    img_path=img_path,
                    when=when,
                    theme_raw=theme_raw,
                    filename_raw=filename_raw,
                    prediction=prediction,
                    raw_response=raw_response,
                )
            )

        except Exception as e:
            failed_count += 1
            print(f"  [Error] Failed to process {img_path.name}: {e}")

            if not DRY_RUN:
                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(img_path),
                        "error": str(e),
                    },
                )

    # ------------------------------------------------------------
    # Phase 2: assign folders (merging sparse themes), then
    # move/copy the files.
    # ------------------------------------------------------------
    assignments = assign_theme_folders(analyzed)

    for idx, (record, theme_folder, theme_words) in enumerate(assignments, start=1):
        img_path = record.img_path
        when = record.when

        try:
            try:
                rel = img_path.relative_to(source_path)
            except ValueError:
                rel = img_path

            # The year folder is the only date-based organization;
            # everything below it is thematic.
            year_folder = str(when.year) if when.year is not None else UNKNOWN_YEAR_FOLDER
            dest_folder = target_path / year_folder / theme_folder

            filename = build_filename(
                img_path=img_path,
                theme_words=theme_words,
                filename_raw=record.filename_raw,
            )

            dest_path = choose_destination(
                dest_folder=dest_folder,
                filename=filename,
                planned=planned,
            )

            try:
                shown_destination = dest_path.relative_to(target_path)
            except ValueError:
                shown_destination = dest_path

            print(f"[{idx}/{len(assignments)}] Organizing: {rel}")
            print(f"  Theme folder: {theme_folder}")
            print(f"  Planned destination: {shown_destination}")

            # ----------------------------------------------------
            # Move/copy file.
            # ----------------------------------------------------
            if not DRY_RUN:
                dest_folder.mkdir(parents=True, exist_ok=True)

                if COPY_INSTEAD_OF_MOVE:
                    shutil.copy2(img_path, dest_path)
                    action = "copied"
                else:
                    shutil.move(str(img_path), str(dest_path))
                    action = "moved"

                index[str(img_path)] = str(dest_path)
                save_index(target_path, index)

                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(img_path),
                        "destination": str(dest_path),
                        "action": action,
                        "when": when.__dict__,
                        "theme_folder": theme_folder,
                        "prediction": record.prediction,
                        "raw_response": record.raw_response[:1000],
                    },
                )

                print(f"  -> {action.capitalize()} to: {shown_destination}")

            planned.add(dest_path)
            success_count += 1

        except Exception as e:
            failed_count += 1
            print(f"  [Error] Failed to process {img_path.name}: {e}")

            if not DRY_RUN:
                append_log(
                    target_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source": str(img_path),
                        "error": str(e),
                    },
                )

    print("\nSummary:")
    print(f"  Processed/planned: {success_count}")
    print(f"  Skipped already processed: {skipped_count}")
    print(f"  Failed: {failed_count}")

    if DRY_RUN:
        print("\nDry run complete. No files were moved or copied.")


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

    print(f"Found {len(video_files)} videos in {source_path}.\n")

    if not video_files:
        return

    index = {} if DRY_RUN else load_index(target_path)
    planned: Set[Path] = set()

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, video_path in enumerate(video_files, start=1):
        try:
            rel = video_path.relative_to(source_path)
        except ValueError:
            rel = video_path

        print(f"[{idx}/{len(video_files)}] Video: {rel}")

        source_key = str(video_path)

        if not DRY_RUN and source_key in index:
            existing_destination = Path(index[source_key])
            if existing_destination.exists():
                print(f"  -> Already processed: {existing_destination}")
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

            print(f"  Date token: {when.date_token} (source: {when.source})")
            print(f"  Planned destination: {shown_destination}")

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

                print(f"  -> {action.capitalize()} to: {shown_destination}")

            planned.add(dest_path)
            success_count += 1

        except Exception as e:
            failed_count += 1
            print(f"  [Error] Failed to process video {video_path.name}: {e}")

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

    print("\nVideo summary:")
    print(f"  Processed/planned: {success_count}")
    print(f"  Skipped already processed: {skipped_count}")
    print(f"  Failed: {failed_count}")

    if DRY_RUN:
        print("\nDry run complete. No videos were moved or copied.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    organize_image_library(
        source_dir=SOURCE_DIRECTORY,
        target_dir=OUTPUT_DIRECTORY,
    )

    process_videos(
        source_dir=SOURCE_DIRECTORY,
        target_dir=OUTPUT_DIRECTORY,
    )