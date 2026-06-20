"""verification decide_verdict·thresholds unit tests (API 키 불필요)"""
from structverify.core.schemas import VerdictType


def test_unit_fallback_threshold_match():
    from structverify.verification.verdict_thresholds import classify_error_rate_fallback

    v, conf, need_m = classify_error_rate_fallback(0.05)
    assert v == VerdictType.MATCH
    assert conf == 0.95
    assert need_m is False


def test_unit_fallback_threshold_unverifiable_mid():
    from structverify.verification.verdict_thresholds import classify_error_rate_fallback

    v, conf, need_m = classify_error_rate_fallback(0.15)
    assert v == VerdictType.UNVERIFIABLE
    assert conf == 0.4
    assert need_m is False


def test_unit_fallback_threshold_mismatch():
    from structverify.verification.verdict_thresholds import classify_error_rate_fallback

    v, conf, need_m = classify_error_rate_fallback(0.50)
    assert v == VerdictType.MISMATCH
    assert need_m is True


def test_unit_fallback_threshold_high_unverifiable():
    from structverify.verification.verdict_thresholds import classify_error_rate_fallback

    v, _, need_m = classify_error_rate_fallback(0.95)
    assert v == VerdictType.UNVERIFIABLE
    assert need_m is False


def test_unit_normalize_period_formats():
    from structverify.verification.row_match import normalize_period

    assert normalize_period("202504") == "202504"
    assert normalize_period("2025-04") == "202504"
    assert normalize_period("2025M04") == "202504"
    assert normalize_period("2025") == "2025"
    assert normalize_period("2025Q1") == "2025"


def test_unit_period_matches_ym():
    from structverify.verification.row_match import period_matches_ym

    assert period_matches_ym("2025-04", "202504") is True
    assert period_matches_ym("2025", "202504") is False
    assert period_matches_ym("202503", "202504") is False
