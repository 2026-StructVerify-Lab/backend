"""detection/config.py — Step 3~5 설정 읽기 (config.detection.* + 레거시 fallback).

새 설정은 config.detection 아래에 두고, 기존 최상위 키(candidate_detection 등)는
동작 호환을 위해 fallback으로 유지한다.
"""
from __future__ import annotations

from typing import Any


def _root(config: dict | None) -> dict[str, Any]:
    return config or {}


def _detection(config: dict | None) -> dict[str, Any]:
    block = _root(config).get("detection")
    return block if isinstance(block, dict) else {}


def candidate_detection_config(config: dict | None = None) -> dict[str, Any]:
    """
    candidate detection 설정 병합.

    우선순위: detection.candidate > candidate_detection (레거시)
    """
    det = _detection(config)
    legacy = _root(config).get("candidate_detection") or {}
    if not isinstance(legacy, dict):
        legacy = {}
    candidate = det.get("candidate") or {}
    if not isinstance(candidate, dict):
        candidate = {}
    merged = dict(legacy)
    merged.update(candidate)
    return merged


def domain_packs_dir(config: dict | None = None) -> str:
    det = _detection(config)
    root = _root(config)
    value = det.get("domain_packs_dir") or root.get("domain_packs_dir")
    return value if value else "domain-packs"


def domain_registry_path(config: dict | None = None) -> str:
    det = _detection(config)
    root = _root(config)
    value = det.get("domain_registry_path") or root.get("domain_registry_path")
    return value if value else "domain-packs/registry.yaml"


def claim_min_confidence(config: dict | None = None, *, default: float = 0.7) -> float:
    """claim 채택 최소 confidence. detection.claim > verification (레거시)."""
    det = _detection(config)
    claim = det.get("claim") or {}
    if isinstance(claim, dict) and claim.get("min_confidence") is not None:
        return float(claim["min_confidence"])
    if det.get("min_confidence") is not None:
        return float(det["min_confidence"])
    vconf = _root(config).get("verification") or {}
    if isinstance(vconf, dict) and vconf.get("min_confidence") is not None:
        return float(vconf["min_confidence"])
    return default


def detected_domain(config: dict | None = None, *, default: str = "general") -> str:
    """runtime이 주입한 도메인. detection / 최상위 키 모두 지원."""
    det = _detection(config)
    root = _root(config)
    value = det.get("detected_domain") or root.get("detected_domain")
    return value if value else default


def llm_config(config: dict | None = None) -> dict[str, Any]:
    """LLM 서브설정: config.llm 위에 detection.llm 덮어쓰기."""
    base = dict(_root(config).get("llm") or {})
    det_llm = _detection(config).get("llm") or {}
    if isinstance(det_llm, dict):
        base.update(det_llm)
    return base
