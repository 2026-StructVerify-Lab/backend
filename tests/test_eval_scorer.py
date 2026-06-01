from structverify.core.schemas import Evidence, VerdictType, VerificationResult
from structverify.eval.outcome.scorer import (
    score_case,
    stat_ids_match,
    values_within_tolerance,
)
from structverify.eval.schemas import OutcomeCase
from uuid import uuid4


def test_values_within_tolerance():
    assert values_within_tolerance(100.0, 100.0) is True
    assert values_within_tolerance(101.0, 100.0, rel=0.02) is True
    assert values_within_tolerance(200.0, 100.0) is False


def test_score_case_verdict_match():
    case = OutcomeCase(
        case_id="t1",
        claim_text="x",
        expected_verdict="match",
        official_value=10.0,
    )
    result = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MATCH,
        evidence=Evidence(source_name="KOSIS", official_value=10.0, stat_table_id="DT_A_B"),
    )
    rec = score_case(case, result, schema_mode="oracle")
    assert rec.verdict_correct is True
    assert rec.value_within_tolerance is True
    assert rec.stat_id_match is True
    assert rec.value_ok_verdict_wrong is False


def test_stat_ids_match_normalized():
    assert stat_ids_match("DT_200Y108", "dt_200y108") is True


def test_value_ok_verdict_wrong():
    case = OutcomeCase(
        case_id="t2",
        claim_text="x",
        expected_verdict="match",
        official_value=10.0,
    )
    result = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MISMATCH,
        evidence=Evidence(source_name="KOSIS", official_value=10.0),
    )
    rec = score_case(case, result)
    assert rec.value_within_tolerance is True
    assert rec.value_ok_verdict_wrong is True
