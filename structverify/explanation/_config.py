"""[리팩] Step 9 설정 로드 — explanation/config.yaml (default.yaml 미수정)"""
from __future__ import annotations

from pathlib import Path

import yaml

_EXPLANATION_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def get_explanation_settings(config: dict | None) -> dict:
    """explanation 설정 병합: config.yaml 기본값 ← config['explanation'] override."""
    merged: dict = {}
    if _EXPLANATION_CONFIG_PATH.is_file():
        with open(_EXPLANATION_CONFIG_PATH, encoding="utf-8") as f:
            merged.update(yaml.safe_load(f) or {})
    merged.update((config or {}).get("explanation") or {})
    return merged
