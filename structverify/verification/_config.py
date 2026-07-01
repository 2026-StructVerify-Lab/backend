"""[리팩] Step 8 설정 로드 — verification/config.yaml (default.yaml 미수정)"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

_VERIFICATION_CONFIG_PATH = Path(__file__).parent / "config.yaml"

VerificationProfile = Literal["agent", "fallback"]

_DEFAULT_PROFILE: VerificationProfile = "fallback"


def get_verification_settings(
    config: dict | None,
    profile: VerificationProfile = _DEFAULT_PROFILE,
) -> dict:
    """판정 프로필 설정 병합.

    우선순위 (낮음 → 높음):
      1. verification/config.yaml → profiles[profile]
      2. config['verification']['profiles'][profile]
      3. config['verification'] flat 키 (프로필 네임스페이스 밖 — 하위 호환)
    """
    merged: dict = {}
    if _VERIFICATION_CONFIG_PATH.is_file():
        with open(_VERIFICATION_CONFIG_PATH, encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        profiles = file_cfg.get("profiles") or {}
        merged.update(profiles.get(profile) or {})

    user_cfg = (config or {}).get("verification") or {}
    user_profiles = user_cfg.get("profiles") or {}
    if isinstance(user_profiles.get(profile), dict):
        merged.update(user_profiles[profile])

    # [리팩] 기존 config.verification flat 키 (예: exaggeration_diff_percent) 하위 호환
    for key, value in user_cfg.items():
        if key == "profiles":
            continue
        merged.setdefault(key, value)

    return merged
