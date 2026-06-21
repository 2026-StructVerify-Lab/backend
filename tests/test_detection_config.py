"""tests/test_detection_config.py — detection._config unit tests (API 키 불필요)."""
from structverify.detection._config import (
    candidate_detection_config,
    candidate_llm_label_floor,
    claim_min_confidence,
    claim_worthy_score_floor,
    detection_defaults,
    detected_domain,
    domain_confidence_threshold,
    domain_packs_dir,
    domain_registry_path,
    llm_config,
    model_tier_for,
)


def test_module_config_yaml_loads_candidate_detection():
    defaults = detection_defaults()
    assert defaults["candidate_detection"]["threshold"] == 0.65
    assert defaults["model_tier"]["domain_classify"] == "light"


def test_candidate_detection_legacy_overrides_defaults():
    cfg = {
        "candidate_detection": {"threshold": 0.5, "concurrency": 3},
        "detection": {"candidate_detection": {"threshold": 0.8}},
    }
    merged = candidate_detection_config(cfg)
    assert merged["threshold"] == 0.5
    assert merged["concurrency"] == 3


def test_candidate_detection_defaults_when_no_legacy():
    assert candidate_detection_config({})["threshold"] == 0.65
    assert candidate_detection_config({})["concurrency"] == 2


def test_domain_paths_legacy_root_overrides_defaults():
    assert domain_packs_dir({"domain_packs_dir": "/custom"}) == "/custom"
    assert domain_packs_dir({"detection": {"domain_packs_dir": "/runtime"}}) == "/runtime"
    assert domain_packs_dir({}) == "domain-packs"


def test_domain_confidence_threshold_from_defaults():
    assert domain_confidence_threshold({}) == 0.6


def test_claim_min_confidence_legacy_verification_wins():
    assert claim_min_confidence({"verification": {"min_confidence": 0.6}}) == 0.6
    assert claim_min_confidence({
        "verification": {"min_confidence": 0.6},
        "detection": {"claim": {"min_confidence": 0.9}},
    }) == 0.6
    assert claim_min_confidence({}) == 0.7


def test_detected_domain_reads_root_and_detection_block():
    assert detected_domain({"detected_domain": "economy"}) == "economy"
    assert detected_domain({
        "detected_domain": "economy",
        "detection": {"detected_domain": "health"},
    }) == "health"


def test_model_tier_from_defaults():
    assert model_tier_for({}, "domain_classify") == "light"
    assert model_tier_for({}, "claim_worthiness") == "heavy"


def test_score_floors_from_defaults():
    assert claim_worthy_score_floor({}) == 0.8
    assert candidate_llm_label_floor({}) == 0.75


def test_llm_config_merges_detection_override():
    cfg = {
        "llm": {"provider": "hcx", "temperature": 0.1},
        "detection": {"llm": {"temperature": 0.3}},
    }
    assert llm_config(cfg) == {"provider": "hcx", "temperature": 0.3}
