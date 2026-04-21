"""
core/config_loader.py — YAML 설정 로더
"""
from __future__ import annotations
import os
from pathlib import Path

def load_config(path: str | None = None) -> dict:
    """config/default.yaml을 로드한다."""
    import yaml
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "config" / "default.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
