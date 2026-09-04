@echo off
setlocal

REM ============================================================
REM Setup for media-classifier
REM
REM Creates a local virtual environment (.venv) inside the repo,
REM installs Python dependencies into it, and downloads the
REM OpenVINO model into .\models\Qwen2.5-VL-7B-Instruct-int4-ov
REM
REM Nothing is installed system-wide and nothing is deleted.
REM Safe to re-run: existing steps are simply skipped or updated.
REM ============================================================

REM --- Check that Python is available ------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [Error] Python was not found on PATH.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and make sure "Add Python to PATH" is checked during install.
    exit /b 1
)

REM --- Create virtual environment if it does not exist --------
if exist ".venv\Scripts\python.exe" (
    echo [Info] Virtual environment already exists, reusing .venv
) else (
    echo [Info] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [Error] Failed to create virtual environment.
        exit /b 1
    )
)

REM --- Activate virtual environment ---------------------------
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [Error] Failed to activate virtual environment.
    exit /b 1
)

REM --- Upgrade pip and install dependencies ------------------
echo [Info] Upgrading pip ...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [Error] Failed to upgrade pip.
    exit /b 1
)

echo [Info] Installing Python dependencies from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo [Error] Failed to install dependencies.
    exit /b 1
)

REM --- Download the OpenVINO model (about 5 GB) -------------
if exist "models\Qwen2.5-VL-7B-Instruct-int4-ov\openvino_language_model.xml" (
    echo [Info] Model already downloaded, skipping download.
) else (
    echo [Info] Downloading OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov ...
    echo [Info] This is about 5 GB and may take a while.

    REM Keep the Hugging Face cache inside the repo as well,
    REM so nothing is written outside the project folder.
    set "HF_HOME=%~dp0models\hf_cache"

    REM Prefer the new "hf" CLI; fall back to the legacy
    REM "huggingface-cli" for older huggingface_hub versions.
    where hf >nul 2>nul
    if not errorlevel 1 (
        hf download OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov --local-dir "models\Qwen2.5-VL-7B-Instruct-int4-ov"
    ) else (
        huggingface-cli download OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov --local-dir "models\Qwen2.5-VL-7B-Instruct-int4-ov"
    )

    if errorlevel 1 (
        echo [Error] Model download failed.
        echo The venv and Python packages are installed, but the model is missing.
        echo Re-run this script to retry the download.
        exit /b 1
    )
)

echo.
echo ============================================================
echo Setup complete.
echo.
echo Activate the environment and run the classifier with:
echo     .venv\Scripts\activate
echo     python main.py
echo ============================================================

endlocal
