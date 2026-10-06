# Media Classifier

A local, offline tool that organizes a folder of unsorted photos and videos into a
clean, event-based library — using a vision-language model (Qwen2.5-VL via OpenVINO)
to describe and name your images automatically.

```
unorganized_images/  ->  organized_library/
    anything.jpg             2023/
    trip/movie.mp4               07 Family Beach Vacation/
    ...                              20230715_183045_sunset-over-rocky-pier.jpg
                                    20230715_183045_sunset-over-rocky-pier.xmp
                                  2022/
                                    ...
                             Videos/
                               2023/
                                 trip/movie.mp4
```

## What it does

**Images** are sorted into `organized_library/<Year>/<MM Event Name>/`.
Filenames start with the EXIF date and time (ISO 8601 basic format), so files
sort chronologically inside every folder, followed by a short content
description:

```
YYYYMMDD_HHMMSS_subject-setting[-action][_NNN].ext
e.g. 20230715_183045_sunset-over-rocky-pier.jpg
     20230715_183050_sunset-over-rocky-pier_002.jpg   (burst photo)
```

The pipeline runs in phases:

1. **Deterministic metadata** — EXIF timestamp, GPS coordinates, and camera
   model are read first, without any AI. GPS points are reverse-geocoded
   offline (city/country) from a local GeoNames database.
2. **Event clustering** — every image is embedded with a small CLIP model and
   clustered with sibling photos from the same year and month. One
   vision-language call per cluster, viewing a contact sheet of its frames,
   names the event ("05 Newborn Photos"). This gives the model the context a
   single photo lacks and keeps naming consistent across an occasion.
3. **Structured description** — one vision-language call per image returns
   structured slots (category, subject, setting, action, details). The
   filename and folder are **composed in code** from those slots — never
   taken verbatim from the model, and words already used by the event name
   are removed from the per-image description.
4. **XMP metadata sidecar** — every image gets a standards-compliant XMP
   sidecar (`.xmp`, ISO 16684-1) with Dublin Core and IPTC Extension
   metadata: title, description, keywords, location, and provenance. Image
   bytes are never modified.

- The **year** is determined in priority order: EXIF date → full date in the
  filename → full date in a parent folder → year in the filename → year in a
  parent folder → (optional) a year the model sees in the image → file
  modification time → `Unknown Year`. These filename/path fallbacks only
  affect the year, never the description or event name.
- The original filename is never used for naming — if the model output is
  unusable, a generic fallback is used instead.

**Videos** are moved (no AI involved) into `organized_library/Videos/<Year>/`,
optionally preserving their original subfolder structure.

Additional features:

- Dry-run mode (`DRY_RUN`) to preview events and planned moves without
  touching files.
- Copy-instead-of-move mode (`COPY_INSTEAD_OF_MOVE`) for safer first runs.
- Resume support: a `_organize_index.json` file remembers already-processed
  files, and an `_analysis_checkpoint.json` file caches AI results so an
  interrupted run only infers the remaining images.
- A `_organize_log.jsonl` audit log of every action taken.
- Runs on the integrated Intel GPU, then CPU (the NPU and discrete GPU are
  excluded; see the comments in `config.py` for why).

## Requirements

- **Windows** (the setup script is a `.bat` file)
- **Python 3.10 or newer** — from [python.org](https://www.python.org/downloads/)
  (check *"Add Python to PATH"* during install)
- **~12 GB free disk space** — about 5 GB for the vision-language model,
  ~0.7 GB for the CLIP model and the offline geocoding database, plus
  Python packages (PyTorch + OpenVINO) in the virtual environment
- **Recommended: 8 GB or more RAM** for the int4-quantized model
- **Optional: ffmpeg** — see the section below

Everything is installed **inside this repository** — the Python virtual
environment lives in `.venv/`, the AI model in `models/`, and the Hugging Face
download cache in `models/hf_cache/`. Nothing is installed system-wide and no
system settings are changed.

## Setup

### 1. Run the setup script

Double-click `setup.bat`, or run it from a terminal in this folder:

```
setup.bat
```

The script will:

1. Create a local virtual environment in `.venv\` (skipped if it already exists).
2. Install the Python dependencies from `requirements.txt` into it.
3. Download the `OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov` model (about 5 GB)
   into `models\`.

It is safe to re-run at any time — it never deletes anything and skips steps
that are already complete.

### 2. Do I need to install ffmpeg separately?

**Short answer: it's optional but recommended.**

The script uses `ffprobe` (part of ffmpeg) to read the embedded creation date
of video files. This is by far the most reliable way to date videos.

- **Without ffmpeg**, the script still works — it falls back to dates found in
  the video filename, parent folder names, or the file modification time.
- **With ffmpeg**, videos dated only by their metadata are sorted correctly
  too.

ffmpeg cannot be installed inside the project's virtual environment (it is a
system tool, not a Python package), so it needs a one-time system install:

**Option A — winget (easiest, on Windows 10/11):**
open a terminal (`Win` → type `cmd`) and run:

```
winget install --id Gyan.FFmpeg
```

**Option B — manual download:**
1. Download a build from <https://www.gyan.dev/ffmpeg/builds/> (e.g. the
   "release full build" `.7z`) or <https://github.com/BtbN/FFmpeg-Builds/releases>.
2. Extract it, e.g. to `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your `PATH`:
   - Press `Win`, type *"environment variables"*, open **Edit the system
     environment variables** → **Environment Variables…**
   - Under *User variables*, select `Path` → **Edit** → **New** → enter
     `C:\ffmpeg\bin` → confirm with **OK**.

**Verify the install** (in a *new* terminal):

```
ffmpeg -version
ffprobe -version
```

## Usage

1. Put the images and videos you want to organize into the
   `unorganized_images/` folder (create it if it doesn't exist; subfolders are
   scanned recursively).
2. Activate the virtual environment:

   ```
   .venv\Scripts\activate
   ```

3. Run:

   ```
   python main.py
   ```

4. Find your organized library in `organized_library/`.

**Tip for a first run:** set `DRY_RUN = True` at the top of
[`config.py`](config.py) to preview every planned move without touching any
files, then set it back to `False` (and optionally
`COPY_INSTEAD_OF_MOVE = True` for a conservative first real run).

## Code layout

The tool is a set of flat modules, run with `python main.py`:

| Module | Purpose |
| --- | --- |
| [`main.py`](main.py) | Entry point |
| [`config.py`](config.py) | All settings; environment bootstrap (HF cache, HEIC support) |
| [`ui.py`](ui.py) | Aesthetic terminal layer; exports the styled `print` |
| [`records.py`](records.py) | `WhenInfo` / `ImageFacts` / `PlannedImage` dataclasses |
| [`helpers.py`](helpers.py) | Word cleanup, year validation, stable hashing, ETA printing |
| [`facts.py`](facts.py) | EXIF time/GPS/camera, filename/path dates, offline geocoding |
| [`vlm.py`](vlm.py) | Qwen model loading, prompts, structured-slot inference |
| [`clustering.py`](clustering.py) | CLIP embeddings, event clustering, contact-sheet naming |
| [`naming.py`](naming.py) | Filename/folder composition (pure code from slots) |
| [`xmp.py`](xmp.py) | XMP sidecar writer (ISO 16684-1, Dublin Core, IPTC) |
| [`state.py`](state.py) | Move index, audit log, analysis checkpoint |
| [`organizer.py`](organizer.py) | The image pipeline (scan → enrich → cluster → name → move) |
| [`videos.py`](videos.py) | Video dating and moving (no AI) |

## Configuration

All settings are constants in [`config.py`](config.py):

| Setting | Default | Meaning |
| --- | --- | --- |
| `SOURCE_DIRECTORY` | `./unorganized_images` | Folder to scan |
| `OUTPUT_DIRECTORY` | `./organized_library` | Destination folder |
| `DRY_RUN` | `False` | Preview only, no files moved/copied |
| `COPY_INSTEAD_OF_MOVE` | `False` | Copy instead of move (safer) |
| `DEVICE_PRIORITY` | `["GPU", "CPU"]` | OpenVINO devices tried in order (integrated GPU, then CPU) |
| `EMBED_MODEL_ID` | `openai/clip-vit-base-patch32` | CLIP model used for event clustering (CPU, milliseconds per image) |
| `CLUSTER_DISTANCE_THRESHOLD` | `0.12` | Cosine distance at which event clustering stops merging — raise to merge more, lower to split |
| `PROCESS_VIDEOS` | `True` | Also organize videos |
| `VIDEO_PRESERVE_RELATIVE_FOLDERS` | `False` | Keep source subfolders under `Videos/<Year>/` |
| `USE_FFPROBE_DATE` | `True` | Use ffmpeg metadata for video dates |
| `SKIP_HIDDEN` | `True` | Ignore hidden files/folders (`.git`, `.DS_Store`, …) |
| `USE_FILESYSTEM_DATE_FALLBACK` | `True` | Fall back to file modification time |
| `ALLOW_MODEL_YEAR_FALLBACK` | `True` | Let the AI guess a year from image content (can be wrong) |
| `ANALYSIS_CHECKPOINT` | `True` | Cache AI results so interrupted runs resume without re-inferring |

## Notes

- **Models:** uses [`OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov`](https://huggingface.co/OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov)
  (int4-quantized 7B vision-language model) for description and event
  naming, and [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
  for event clustering. Local copies under `models/` are used when present;
  otherwise they are downloaded from Hugging Face on first run.
- **Geocoding:** the `reverse_geocoder` package downloads a ~85 MB GeoNames
  database on first use. Everything stays offline. If it is unavailable,
  geocoding silently no-ops and raw coordinates are still recorded in the
  XMP sidecars.
- **XMP sidecars:** each image gets a `.xmp` file with `dc:title`,
  `dc:description`, keywords, IPTC `LocationCreated`, and a `mcs:`
  provenance block containing the original path and the full AI
  prediction. Any XMP-aware tool (Lightroom, digiKam, …) reads these.
- **Resume:** delete `_organize_index.json` in the output folder to force
  re-processing of everything. Checkpointed AI results are keyed by file
  size and modification time, so edited files are re-analyzed.
- **Re-running after moving files:** the index keys on the original source
  path, so re-running with `COPY_INSTEAD_OF_MOVE = True` after deleting the
  index can create duplicates — keep the index intact.
- The first run is slow while the model loads (about 5 GB into RAM/VRAM);
  each image then takes roughly 10–20 s on the integrated GPU, longer on CPU.
