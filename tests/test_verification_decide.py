"""verification decide_verdict·thresholds unit tests (API 키 불필요)"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from structverify.agent.schemas import ActionType, ClaimType, Plan
from structverify.core.schemas import (
    Claim,
    ClaimSchema,
    Evidence,
    SourceOffset,
    VerdictType,
)


def _claim(
    *,
    claim_text: str = "test",
    value: float | None = 1.0,
    unit: str | None = None,
    time_period: str | None = None,
    indicator: str | None = None,
    prev_value: float | None = None,
    prev_time_period: str | None = None,
    modifier: str | None = None,
) -> Claim:
    return Claim(
        doc_id=uuid4(),
        block_id="b0",
        sent_id="s0",
        claim_text=claim_text,
        schema=ClaimSchema(
            value=value,
            unit=unit,
            time_period=time_period,
            indicator=indicator,
            prev_value=prev_value,
            prev_time_period=prev_time_period,
            modifier=modifier,
            parent_path="",
        ),
        source_offset=SourceOffset(),
    )


def _kosis_rows(*rows: dict) -> dict:
    return {"row": list(rows)}


def _evidence(
    *,
    official_value: float | None = None,
    unit: str = "",
    time_period: str = "",
    raw_response: dict | None = None,
) -> Evidence:
    return Evidence(
        source_name="KOSIS",
        stat_table_id="DT_TEST",
        official_value=official_value,
        unit=unit,
        time_period=time_period,
        raw_response=raw_response or {},
    )


def _agent_fetch_input(
    claim: Claim,
    evidence: dict,
    *,
    claim_type: ClaimType = ClaimType.ABSOLUTE,
    tolerance: float = 0.05,
) -> "AgentFetchInput":
    from structverify.verification.adapters import AgentFetchInput

    return AgentFetchInput(
        claim_id=str(claim.claim_id),
        evidence=evidence,
        claim_actual_type=claim_type,
        plan_claim_type=claim_type,
        tolerance=tolerance,
    )


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


def test_unit_infer_claim_type_absolute():
    from uuid import uuid4

    from structverify.agent.schemas import ClaimType
    from structverify.core.schemas import Claim, ClaimSchema, SourceOffset
    from structverify.verification.adapters import infer_claim_type

    claim = Claim(
        doc_id=uuid4(),
        block_id="b0",
        sent_id="s0",
        claim_text="농가 166558가구",
        schema=ClaimSchema(value=166558.0, parent_path="", modifier=""),
        source_offset=SourceOffset(),
    )
    assert infer_claim_type(claim) == ClaimType.ABSOLUTE


def test_unit_decide_verdict_agent_fetch_match():
    from uuid import uuid4

    from structverify.agent.schemas import ClaimType
    from structverify.core.schemas import Claim, ClaimSchema, SourceOffset, VerdictType
    from structverify.verification.adapters import AgentFetchInput
    from structverify.verification.decide_verdict import decide_verdict

    claim = Claim(
        doc_id=uuid4(),
        block_id="b0",
        sent_id="s0",
        claim_text="고령화 64.2%",
        schema=ClaimSchema(
            value=64.2,
            time_period="2025-04",
            parent_path="",
            modifier="",
        ),
        source_offset=SourceOffset(),
    )
    normalized = AgentFetchInput(
        claim_id=str(claim.claim_id),
        evidence={
            "value": 64.2,
            "unit": "%",
            "time_period": "202504",
            "stat_table_id": "DT_TEST",
        },
        claim_actual_type=ClaimType.ABSOLUTE,
        plan_claim_type=ClaimType.ABSOLUTE,
        tolerance=0.05,
    )
    result = decide_verdict(claim, normalized, profile="agent")
    assert result.verdict == VerdictType.MATCH
    assert result.confidence == 0.85


# ── fallback 통합 ────────────────────────────────────────────────

def test_fallback_decide_verdict_row_match():
    from structverify.verification.adapters import NormalizedInput
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(value=20171.0, unit="명", time_period="2025-04")
    evidence = _evidence(
        official_value=999.0,
        unit="명",
        time_period="202504",
        raw_response=_kosis_rows(
            {"DT": "20171", "UNIT_NM": "명", "PRD_DE": "202504"},
        ),
    )
    normalized = NormalizedInput(
        evidence=evidence,
        claim_year="2025",
        claim_year_month="202504",
    )
    result = decide_verdict(claim, normalized, profile="fallback")
    assert result.verdict == VerdictType.MATCH


def test_fallback_decide_verdict_growth_auto_calc():
    from structverify.verification.adapters import NormalizedInput
    from structverify.verification.decide_verdict import decide_verdict

    # (20171 - 19059) / 19059 * 100 ≈ 5.83%
    claim = _claim(
        value=5.83,
        unit="%",
        time_period="2025-04",
        indicator="출생아 수 증가율",
        prev_value=19059.0,
    )
    evidence = _evidence(
        official_value=19059.0,
        unit="%",
        time_period="202504",
        raw_response=_kosis_rows(
            {"DT": "20171", "UNIT_NM": "명", "PRD_DE": "202504"},
        ),
    )
    normalized = NormalizedInput(
        evidence=evidence,
        claim_year="2025",
        claim_year_month="202504",
    )
    result = decide_verdict(claim, normalized, profile="fallback")
    assert result.verdict == VerdictType.MATCH


def test_fallback_decide_verdict_difference_auto_calc():
    from structverify.verification.adapters import NormalizedInput
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(
        value=0.04,
        unit="",
        time_period="2025-04",
        indicator="합계출산율 차이",
        prev_value=0.72,
    )
    evidence = _evidence(
        official_value=0.72,
        unit="",
        time_period="202504",
        raw_response=_kosis_rows(
            {"DT": "0.76", "UNIT_NM": "", "PRD_DE": "202504"},
        ),
    )
    normalized = NormalizedInput(
        evidence=evidence,
        claim_year="2025",
        claim_year_month="202504",
    )
    result = decide_verdict(claim, normalized, profile="fallback")
    assert result.verdict == VerdictType.MATCH


def test_verify_claim_row_match_integration():
    from structverify.verification.verifier import verify_claim

    claim = _claim(value=20171.0, unit="명", time_period="2025-04")
    evidence = _evidence(
        official_value=20171.0,
        unit="명",
        time_period="202504",
        raw_response=_kosis_rows(
            {"DT": "20171", "UNIT_NM": "명", "PRD_DE": "202504"},
        ),
    )
    result = verify_claim(claim, evidence)
    assert result.verdict == VerdictType.MATCH


# ── agent 통합 ───────────────────────────────────────────────────

def test_agent_fetch_time_mismatch_unverifiable():
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(value=100.0, unit="%", time_period="2025-04")
    normalized = _agent_fetch_input(
        claim,
        {
            "value": 50.0,
            "unit": "%",
            "time_period": "202401",
            "stat_table_id": "DT_TEST",
        },
    )
    result = decide_verdict(claim, normalized, profile="agent")
    assert result.verdict == VerdictType.UNVERIFIABLE
    assert result.confidence == 0.35


def test_agent_fetch_threshold_gte_match():
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(
        claim_text="고령화율 64% 이상",
        value=64.0,
        unit="%",
        time_period="2025-04",
        modifier="이상",
    )
    normalized = _agent_fetch_input(
        claim,
        {
            "value": 65.0,
            "unit": "%",
            "time_period": "202504",
            "stat_table_id": "DT_TEST",
        },
    )
    result = decide_verdict(claim, normalized, profile="agent")
    assert result.verdict == VerdictType.MATCH
    assert result.confidence == 0.8


def test_agent_fetch_threshold_lte_mismatch():
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(
        claim_text="실업률 3% 이하",
        value=3.0,
        unit="%",
        time_period="2025-04",
        modifier="이하",
    )
    normalized = _agent_fetch_input(
        claim,
        {
            "value": 4.2,
            "unit": "%",
            "time_period": "202504",
            "stat_table_id": "DT_TEST",
        },
    )
    result = decide_verdict(claim, normalized, profile="agent")
    assert result.verdict == VerdictType.MISMATCH


def test_agent_growth_rate_from_rows_match():
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(
        value=5.83,
        unit="%",
        time_period="2025-04",
        indicator="출생아 수 증가율",
        prev_time_period="2024-04",
    )
    normalized = _agent_fetch_input(
        claim,
        {
            "value": 20171,
            "unit": "명",
            "time_period": "202504",
            "stat_table_id": "DT_TEST",
            "matched_row": {"PRD_DE": "202504", "DT": "20171", "ITM_NM": "출생아수"},
            "rows": [
                {"PRD_DE": "202504", "DT": "20171", "ITM_NM": "출생아수"},
                {"PRD_DE": "202404", "DT": "19059", "ITM_NM": "출생아수"},
            ],
        },
        claim_type=ClaimType.GROWTH_RATE,
    )
    result = decide_verdict(claim, normalized, profile="agent")
    assert result.verdict == VerdictType.MATCH


def test_agent_growth_rate_direction_mismatch():
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(
        value=7.7,
        unit="%",
        time_period="2025-04",
        indicator="출생아 수 증가율",
        prev_time_period="2024-04",
    )
    normalized = _agent_fetch_input(
        claim,
        {
            "value": 18000,
            "unit": "명",
            "time_period": "202504",
            "stat_table_id": "DT_TEST",
            "matched_row": {"PRD_DE": "202504", "DT": "18000", "ITM_NM": "출생아수"},
            "rows": [
                {"PRD_DE": "202504", "DT": "18000", "ITM_NM": "출생아수"},
                {"PRD_DE": "202404", "DT": "20171", "ITM_NM": "출생아수"},
            ],
        },
        claim_type=ClaimType.GROWTH_RATE,
    )
    result = decide_verdict(claim, normalized, profile="agent")
    assert result.verdict == VerdictType.MISMATCH
    assert result.confidence == 0.75


def test_agent_calculate_growth_rate_match():
    from structverify.verification.adapters import AgentCalculateInput
    from structverify.verification.decide_verdict import decide_verdict

    claim = _claim(
        value=8.7,
        unit="%",
        time_period="2025-04",
        indicator="혼인 건수 증가율",
    )
    normalized = AgentCalculateInput(
        claim_id=str(claim.claim_id),
        calc_value=8.7,
        claim_actual_type=ClaimType.GROWTH_RATE,
        calc_summary="(current-prev)/prev*100",
    )
    result = decide_verdict(claim, normalized, profile="agent")
    assert result.verdict == VerdictType.MATCH
    assert result.confidence == 0.8


def test_agent_from_calculate_rejects_without_fetch():
    from structverify.verification.adapters import from_agent_calculate

    claim = _claim(
        value=8.7,
        unit="%",
        indicator="혼인 건수 증가율",
        time_period="2025-04",
    )
    plan = Plan(claim_id=str(claim.claim_id), claim_type=ClaimType.GROWTH_RATE, required_data=[])
    calc_obs = SimpleNamespace(
        success=True,
        action=ActionType.CALCULATE,
        input={"current": 100},
        output={"result": 8.7},
        summary="test",
    )
    normalized, early = from_agent_calculate(
        claim, calc_obs, plan, last_fetch_observation=None, workspace=None,
    )
    assert normalized is None
    assert early is None


def test_agent_from_calculate_rejects_non_derived_indicator():
    from structverify.verification.adapters import from_agent_calculate

    claim = _claim(
        value=18921.0,
        unit="건",
        indicator="혼인 건수",
        time_period="2025-04",
    )
    plan = Plan(claim_id=str(claim.claim_id), claim_type=ClaimType.ABSOLUTE, required_data=[])
    calc_obs = SimpleNamespace(
        success=True,
        action=ActionType.CALCULATE,
        input={},
        output={"result": 18921.0},
        summary="test",
    )
    fetch_obs = SimpleNamespace(success=True, action=ActionType.FETCH_EVIDENCE)
    normalized, early = from_agent_calculate(
        claim, calc_obs, plan, last_fetch_observation=fetch_obs,
    )
    assert normalized is None
    assert early is None
