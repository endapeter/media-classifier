# Media Classifier

A local, offline tool that organizes a folder of unsorted photos and videos into a
clean, year-based library — using a vision-language model (Qwen2.5-VL via OpenVINO)
to name and sort your images automatically.

```
unorganized_images/  ->  organized_library/
    anything.jpg             2023/
    trip/movie.mp4               Family Beach Vacation/
    ...                              20230715_family_beach_sunset_on_pier_a1b2c3.jpg
                                  2022/
                                    ...
                             Videos/
                               2023/
                                 trip/movie.mp4
```

## What it does

**Images** are sorted into `organized_library/<Year>/<Thematic Folder>/` with
controlled filenames:

```
{date}_{theme}_{description}_{hash}.ext
e.g. 20230715_family_beach_sunset_on_pier_a1b2c3.jpg
```

- The **date/year** is determined in priority order: EXIF date → full date in the
  filename → full date in a parent folder → year in the filename → year in a
  parent folder → (optional) a year the model sees in the image → file
  modification time → `Unknown Year`.
- The **thematic folder and description** are proposed by the vision-language
  model, then sanitized into safe folder/file names (no illegal Windows
  characters, no years in folder names, length limits, collision-proofing).

**Videos** are moved (no AI involved) into `organized_library/Videos/<Year>/`,
optionally preserving their original subfolder structure.

Additional features:

- Dry-run mode (`DRY_RUN`) to preview without touching files.
- Copy-instead-of-move mode (`COPY_INSTEAD_OF_MOVE`) for safer first runs.
- Resume support: a `_organize_index.json` file remembers already-processed
  files, so interrupted runs can continue where they left off.
- A `_organize_log.jsonl` audit log of every action taken.
- Tries the NPU first (e.g. Intel AI Boost), then the dedicated GPU
  (`GPU.1`, e.g. RTX 4050), then the integrated GPU, then CPU.

## Requirements

- **Windows** (the setup script is a `.bat` file)
- **Python 3.10 or newer** — from [python.org](https://www.python.org/downloads/)
  (check *"Add Python to PATH"* during install)
- **~12 GB free disk space** — about 5 GB for the AI model, plus Python
  packages (PyTorch + OpenVINO) in the virtual environment
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

**Tip for a first run:** set `DRY_RUN = True` at the top of `main.py` to
preview every planned move without touching any files, then set it back to
`False` (and optionally `COPY_INSTEAD_OF_MOVE = True` for a conservative first
real run).

## Configuration

All settings are constants near the top of [`main.py`](main.py):

| Setting | Default | Meaning |
| --- | --- | --- |
| `SOURCE_DIRECTORY` | `./unorganized_images` | Folder to scan |
| `OUTPUT_DIRECTORY` | `./organized_library` | Destination folder |
| `DRY_RUN` | `False` | Preview only, no files moved/copied |
| `COPY_INSTEAD_OF_MOVE` | `False` | Copy instead of move (safer) |
| `DEVICE_PRIORITY` | `["NPU", "GPU.1", "GPU", "CPU"]` | Devices tried in order: NPU (Intel AI Boost), dedicated GPU (`GPU.1`, e.g. RTX 4050), integrated GPU, CPU |
| `FOLDER_STYLE` | `title_spaces` | `title_spaces` → `"Family Beach Vacation"`, `lower_underscores` → `"family_beach_vacation"` |
| `PROCESS_VIDEOS` | `True` | Also organize videos |
| `VIDEO_PRESERVE_RELATIVE_FOLDERS` | `True` | Keep source subfolders under `Videos/<Year>/` |
| `USE_FFPROBE_DATE` | `True` | Use ffmpeg metadata for video dates |
| `SKIP_HIDDEN` | `True` | Ignore hidden files/folders (`.git`, `.DS_Store`, …) |
| `USE_FILESYSTEM_DATE_FALLBACK` | `True` | Fall back to file modification time |
| `ALLOW_MODEL_YEAR_FALLBACK` | `True` | Let the AI guess a year from image content (can be wrong) |

## Notes

- **Model:** uses [`OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov`](https://huggingface.co/OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov),
  an int4-quantized 7B vision-language model. If `models/Qwen2.5-VL-7B-Instruct-int4-ov/`
  exists, it is loaded locally; otherwise it is downloaded from Hugging Face on
  first run.
- **Resume:** delete `_organize_index.json` in the output folder to force
  re-processing of everything.
- **Re-running after moving files:** the index keys on the original source
  path, so re-running with `COPY_INSTEAD_OF_MOVE = True` after deleting the
  index can create duplicates — keep the index intact.
- The first run is slow while the model loads (about 5 GB into RAM/VRAM);
  each image then takes roughly a few seconds on GPU, longer on CPU.
