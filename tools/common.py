from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_git_branch(default: str = "unknown") -> str:
    branch = os.getenv("SV_BRANCH")
    if branch:
        return branch
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return value or default
    except Exception:
        return default


def get_run_version(default: str = "v001") -> str:
    return os.getenv("SV_VERSION", default)


def get_run_dir() -> Path:
    base = Path(os.getenv("SV_RESULTS_DIR", "./run_outputs"))
    run_dir = base / f"{get_git_branch()}_{get_run_version()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def should_save() -> bool:
    return os.getenv("SV_SAVE_RESULTS", "1") not in {"0", "false", "False", "no", "NO"}


def serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serialize(v) for v in obj]
    return obj


def save_json(payload: Any, name: str, metadata: dict[str, Any] | None = None) -> Path | None:
    if not should_save():
        return None
    run_dir = get_run_dir()
    path = run_dir / f"{name}.json"
    wrapped = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "branch": get_git_branch(),
        "version": get_run_version(),
        "metadata": metadata or {},
        "payload": serialize(payload),
    }
    path.write_text(json.dumps(wrapped, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
