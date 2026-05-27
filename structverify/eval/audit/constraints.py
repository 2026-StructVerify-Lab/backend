"""Post-hoc constraint checks on verification results."""
from __future__ import annotations

from typing import Any

from structverify.core.schemas import VerdictType, VerificationResult
from structverify.eval.outcome.scorer import values_within_tolerance


def check_constraints(
    result: VerificationResult,
    *,
    stated_value: float | None = None,
    official_value: float | None = None,
    rel: float = 0.005,
    abs_tol: float = 0.1,
) -> list[dict[str, str]]:
    """Return list of violation dicts (rule_id, message). Empty if OK."""
    violations: list[dict[str, str]] = []
    v = result.verdict
    ev = result.evidence

    if v in (VerdictType.MATCH, VerdictType.MISMATCH) and ev is None:
        violations.append({"rule_id": "R1", "message": "verdict set but no evidence"})

    if ev is not None:
        if ev.official_value is not None and not ev.stat_table_id:
            violations.append(
                {"rule_id": "R2", "message": "official_value without stat_table_id"}
            )

    if (
        v == VerdictType.MATCH
        and ev is not None
        and ev.official_value is not None
        and stated_value is not None
        and official_value is not None
    ):
        ok = values_within_tolerance(
            ev.official_value, official_value, rel=rel, abs_tol=abs_tol
        )
        if ok is False:
            violations.append(
                {
                    "rule_id": "R3",
                    "message": "match but evidence value outside gold tolerance",
                }
            )

    return violations


def audit_result_row(
    case_id: str,
    result: VerificationResult | None,
    *,
    stated_value: float | None = None,
    official_value: float | None = None,
    rel: float = 0.005,
    abs_tol: float = 0.1,
) -> dict[str, Any]:
    if result is None:
        return {
            "case_id": case_id,
            "constraint_violations": [{"rule_id": "R0", "message": "no result"}],
        }
    violations = check_constraints(
        result,
        stated_value=stated_value,
        official_value=official_value,
        rel=rel,
        abs_tol=abs_tol,
    )
    return {
        "case_id": case_id,
        "predicted_verdict": result.verdict.value,
        "constraint_violations": violations,
        "has_violation": bool(violations),
    }
