"""[리팩] Step 3~5 설정 로드 — detection/config.yaml (default.yaml 미수정).

런타임: config.detection.*
레거시: candidate_detection, verification.min_confidence 등 (fallback, 변경 없음)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).with_name("config.yaml")
_CONFIG_CACHE: dict[str, Any] | None = None


def _load_module_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _CONFIG_CACHE = data if isinstance(data, dict) else {}
    return dict(_CONFIG_CACHE)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _root(config: dict | None) -> dict[str, Any]:
    return config or {}


def _runtime_detection(config: dict | None) -> dict[str, Any]:
    block = _root(config).get("detection")
    return block if isinstance(block, dict) else {}


def detection_defaults() -> dict[str, Any]:
    """detection/config.yaml 내용 (테스트·디버그용)."""
    return _load_module_config()


def _detection_effective(config: dict | None) -> dict[str, Any]:
    """config.yaml + config.detection 병합 (레거시 키 제외)."""
    return _deep_merge(_load_module_config(), _runtime_detection(config))


def candidate_detection_config(config: dict | None = None) -> dict[str, Any]:
    """
    candidate detection 설정 병합.

    우선순위: candidate_detection (레거시) > config.detection > config.yaml
    """
    det = _detection_effective(config)
    legacy = _root(config).get("candidate_detection") or {}
    if not isinstance(legacy, dict):
        legacy = {}
    base = det.get("candidate_detection") or det.get("candidate") or {}
    if not isinstance(base, dict):
        base = {}
    merged = dict(base)
    merged.update(legacy)
    return merged


def domain_packs_dir(config: dict | None = None) -> str:
    root = _root(config)
    if root.get("domain_packs_dir"):
        return str(root["domain_packs_dir"])
    runtime = _runtime_detection(config)
    if runtime.get("domain_packs_dir"):
        return str(runtime["domain_packs_dir"])
    return str(_load_module_config().get("domain_packs_dir", "domain-packs"))


def domain_registry_path(config: dict | None = None) -> str:
    root = _root(config)
    if root.get("domain_registry_path"):
        return str(root["domain_registry_path"])
    runtime = _runtime_detection(config)
    if runtime.get("domain_registry_path"):
        return str(runtime["domain_registry_path"])
    return str(_load_module_config().get("domain_registry_path", "domain-packs/registry.yaml"))


def domain_confidence_threshold(config: dict | None = None) -> float:
    det = _detection_effective(config)
    domain = det.get("domain") or {}
    if isinstance(domain, dict) and domain.get("confidence_threshold") is not None:
        return float(domain["confidence_threshold"])
    if det.get("confidence_threshold") is not None:
        return float(det["confidence_threshold"])
    return 0.6


def claim_min_confidence(config: dict | None = None) -> float:
    """claim 채택 최소 confidence. 레거시 verification.min_confidence fallback."""
    root = _root(config)
    vconf = root.get("verification") or {}
    if isinstance(vconf, dict) and vconf.get("min_confidence") is not None:
        return float(vconf["min_confidence"])
    runtime = _runtime_detection(config)
    claim = runtime.get("claim") or {}
    if isinstance(claim, dict) and claim.get("min_confidence") is not None:
        return float(claim["min_confidence"])
    if runtime.get("min_confidence") is not None:
        return float(runtime["min_confidence"])
    defaults_claim = _load_module_config().get("claim") or {}
    if isinstance(defaults_claim, dict) and defaults_claim.get("min_confidence") is not None:
        return float(defaults_claim["min_confidence"])
    return 0.7


def claim_worthy_score_floor(config: dict | None = None) -> float:
    """check-worthy true인데 score=0일 때 보정값."""
    det = _detection_effective(config)
    claim = det.get("claim") or {}
    if isinstance(claim, dict) and claim.get("worthy_score_floor") is not None:
        return float(claim["worthy_score_floor"])
    return 0.8


def candidate_llm_label_floor(config: dict | None = None) -> float:
    det = _detection_effective(config)
    cand = det.get("candidate_detection") or det.get("candidate") or {}
    if isinstance(cand, dict) and cand.get("llm_label_floor_score") is not None:
        return float(cand["llm_label_floor_score"])
    return 0.75


def model_tier_for(config: dict | None, step: str, *, default: str = "heavy") -> str:
    """Step별 LLM model_tier (domain_classify, candidate_score, ...)."""
    det = _detection_effective(config)
    tiers = det.get("model_tier") or {}
    if isinstance(tiers, dict) and tiers.get(step):
        return str(tiers[step])
    return default


def detected_domain(config: dict | None = None, *, default: str = "general") -> str:
    det = _detection_effective(config)
    root = _root(config)
    value = det.get("detected_domain") or root.get("detected_domain")
    return value if value else default


def llm_config(config: dict | None = None) -> dict[str, Any]:
    base = dict(_root(config).get("llm") or {})
    det_llm = _detection_effective(config).get("llm") or {}
    if isinstance(det_llm, dict):
        base.update(det_llm)
    return base
