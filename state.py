"""
Logging and resume state: the move index, the audit log, and the analysis
checkpoint.
"""

import json
from pathlib import Path
from typing import Any, Dict

from config import CHECKPOINT_SCHEMA
from ui import print


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
# Analysis checkpoint
# ============================================================

def load_checkpoint(target_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the analysis checkpoint: source path -> VLM analysis.

    Unlike the move index, this is written during analysis, before any file
    is moved, so an interrupted run can resume without re-running inference.
    """
    checkpoint_path = target_path / "_analysis_checkpoint.json"

    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return {}

    return {}


def save_checkpoint(target_path: Path, checkpoint: Dict[str, Dict[str, Any]]) -> None:
    checkpoint_path = target_path / "_analysis_checkpoint.json"

    try:
        target_path.mkdir(parents=True, exist_ok=True)

        tmp_path = checkpoint_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(checkpoint_path)

    except Exception as e:
        print(f"  [Warning] Could not save analysis checkpoint: {e}")


def checkpoint_matches_file(img_path: Path, entry: Dict[str, Any]) -> bool:
    """
    True if the checkpoint entry still describes this exact file.

    Guards against reusing an analysis for a file that was replaced or
    edited after being analyzed, and rejects entries written by an older
    schema (no structured slots) so those images are re-analyzed.
    """
    try:
        st = img_path.stat()
    except OSError:
        return False

    return (
        entry.get("schema") == CHECKPOINT_SCHEMA
        and isinstance(entry.get("slots"), dict)
        and entry.get("size") == st.st_size
        and entry.get("mtime_ns") == st.st_mtime_ns
    )
