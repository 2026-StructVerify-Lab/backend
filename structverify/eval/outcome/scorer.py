"""Score outcome predictions vs gold labels."""
from __future__ import annotations

from structverify.eval.schemas import OutcomeCase, OutcomePredictionRecord
from structverify.core.schemas import VerificationResult


def values_within_tolerance(
    predicted: float | None,
    official: float | None,
    *,
    rel: float = 0.005,
    abs_tol: float = 0.1,
) -> bool | None:
    if predicted is None or official is None:
        return None
    if official == 0:
        return abs(predicted - official) <= abs_tol
    return abs(predicted - official) <= max(abs_tol, abs(official) * rel)


def score_case(
    case: OutcomeCase,
    result: VerificationResult | None,
    *,
    rel: float = 0.005,
    abs_tol: float = 0.1,
    error: str | None = None,
) -> OutcomePredictionRecord:
    if error or result is None:
        return OutcomePredictionRecord(
            case_id=case.case_id,
            expected_verdict=case.expected_verdict,
            error=error or "no_result",
        )
    pred_v = result.verdict.value
    verdict_correct = pred_v == case.expected_verdict
    pred_official = None
    pred_stat = None
    if result.evidence:
        pred_official = result.evidence.official_value
        pred_stat = result.evidence.stat_table_id
    value_ok = values_within_tolerance(
        pred_official,
        case.official_value,
        rel=rel,
        abs_tol=abs_tol,
    )
    return OutcomePredictionRecord(
        case_id=case.case_id,
        expected_verdict=case.expected_verdict,
        predicted_verdict=pred_v,
        verdict_correct=verdict_correct,
        value_within_tolerance=value_ok,
        predicted_official_value=pred_official,
        reference_stat_id=case.reference_stat_id,
        predicted_stat_id=pred_stat,
    )
