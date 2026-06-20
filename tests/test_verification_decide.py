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


def test_unit_adapter_from_evidence_no_evidence():
    from uuid import uuid4

    from structverify.core.schemas import Claim, SourceOffset, VerdictType
    from structverify.verification.adapters import from_evidence

    claim = Claim(
        doc_id=uuid4(),
        block_id="b0",
        sent_id="s0",
        claim_text="test",
        source_offset=SourceOffset(),
    )
    normalized, early = from_evidence(claim, None)
    assert normalized is None
    assert early is not None
    assert early.verdict == VerdictType.UNVERIFIABLE


def test_unit_agent_atomic_threshold_match():
    from structverify.verification.verdict_thresholds import classify_atomic_ratio_agent

    v, conf = classify_atomic_ratio_agent(0.04)
    assert v == VerdictType.MATCH
    assert conf == 0.85


def test_unit_agent_growth_rate_pp_match():
    from structverify.verification.verdict_thresholds import classify_growth_rate_pp_agent

    v, _ = classify_growth_rate_pp_agent(1.0)
    assert v == VerdictType.MATCH


def test_unit_agent_difference_gap_match():
    from structverify.verification.verdict_thresholds import classify_difference_gap_agent

    v, _ = classify_difference_gap_agent(0.01, 0.06)
    assert v == VerdictType.MATCH


def test_unit_row_match_find_value_for_time():
    from structverify.verification.row_match import find_value_for_time_with_criteria

    rows = [{"PRD_DE": "202504", "DT": "20171", "ITM_NM": "출생아수"}]
    hit = find_value_for_time_with_criteria(rows, "2025-04", {"ITM_NM": "출생아수"})
    assert hit is not None
    assert hit[0] == 20171.0
