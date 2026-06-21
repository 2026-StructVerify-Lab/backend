"""tests/test_detection_config.py — detection.config unit tests (API 키 불필요)."""
from structverify.detection.config import (
    candidate_detection_config,
    claim_min_confidence,
    detected_domain,
    domain_packs_dir,
    domain_registry_path,
    llm_config,
)


def test_candidate_detection_legacy_only():
    cfg = {"candidate_detection": {"threshold": 0.5, "concurrency": 3}}
    assert candidate_detection_config(cfg) == {"threshold": 0.5, "concurrency": 3}


def test_candidate_detection_new_overrides_legacy():
    cfg = {
        "candidate_detection": {"threshold": 0.5, "concurrency": 3},
        "detection": {"candidate": {"threshold": 0.8}},
    }
    merged = candidate_detection_config(cfg)
    assert merged["threshold"] == 0.8
    assert merged["concurrency"] == 3


def test_domain_paths_fallback():
    assert domain_packs_dir({}) == "domain-packs"
    assert domain_packs_dir({"detection": {"domain_packs_dir": "/custom"}}) == "/custom"
    assert domain_registry_path({}) == "domain-packs/registry.yaml"


def test_claim_min_confidence_priority():
    assert claim_min_confidence({"verification": {"min_confidence": 0.6}}) == 0.6
    assert claim_min_confidence({
        "verification": {"min_confidence": 0.6},
        "detection": {"claim": {"min_confidence": 0.9}},
    }) == 0.9


def test_detected_domain_reads_root_and_detection_block():
    assert detected_domain({"detected_domain": "economy"}) == "economy"
    assert detected_domain({
        "detected_domain": "economy",
        "detection": {"detected_domain": "health"},
    }) == "health"


def test_llm_config_merges_detection_override():
    cfg = {
        "llm": {"provider": "hcx", "temperature": 0.1},
        "detection": {"llm": {"temperature": 0.3}},
    }
    assert llm_config(cfg) == {"provider": "hcx", "temperature": 0.3}
