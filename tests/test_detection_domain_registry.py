"""tests/test_detection_domain_registry.py — DomainRegistry unit tests."""
import yaml

from structverify.detection.domain.registry import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_SEED_DOMAINS,
    DOMAIN_NAME_PATTERN,
    DomainRegistry,
)


def test_confidence_threshold_constant():
    assert CONFIDENCE_THRESHOLD == 0.6


def test_domain_name_pattern_accepts_valid_keys():
    assert DOMAIN_NAME_PATTERN.match("real_estate")
    assert not DOMAIN_NAME_PATTERN.match("Real-Estate")


def test_registry_load_creates_seed_file_when_missing(tmp_path):
    path = tmp_path / "registry.yaml"
    registry = DomainRegistry(str(path))
    loaded = registry.load()
    assert loaded == DEFAULT_SEED_DOMAINS
    assert path.exists()


def test_registry_register_persists_new_domain(tmp_path):
    path = tmp_path / "registry.yaml"
    registry = DomainRegistry(str(path))
    registry.load()
    registry.register("real_estate", "부동산 통계")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["real_estate"] == "부동산 통계"
