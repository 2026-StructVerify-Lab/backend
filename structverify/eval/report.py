"""Aggregate 3-axis eval metrics into a single report."""
from __future__ import annotations

from typing import Any


def _prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_outcome(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    if n == 0:
        return {"verdict_accuracy": 0.0, "value_tolerance_rate": 0.0, "n": 0}
    verdict_ok = sum(1 for p in predictions if p.get("verdict_correct"))
    value_checked = [p for p in predictions if p.get("value_within_tolerance") is not None]
    value_ok = sum(1 for p in value_checked if p.get("value_within_tolerance"))
    errors = sum(1 for p in predictions if p.get("error"))
    return {
        "n": n,
        "verdict_accuracy": verdict_ok / n,
        "value_tolerance_rate": value_ok / len(value_checked) if value_checked else None,
        "errors": errors,
        "verdict_correct": verdict_ok,
    }


def summarize_audit(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(audit_rows)
    if n == 0:
        return {"constraint_violation_rate": 0.0, "kosis_grounding_rate": 0.0, "n": 0}
    violations = sum(1 for r in audit_rows if r.get("constraint_violations"))
    grounded = sum(1 for r in audit_rows if r.get("kosis_grounding_ok"))
    checked = sum(1 for r in audit_rows if r.get("kosis_grounding_checked"))
    return {
        "n": n,
        "constraint_violation_rate": violations / n,
        "kosis_grounding_rate": grounded / checked if checked else None,
        "violations": violations,
        "grounded": grounded,
        "grounding_checked": checked,
    }


def summarize_detection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for r in rows:
        exp = r.get("should_extract")
        got = r.get("predicted_extract")
        if exp and got:
            tp += 1
        elif exp and not got:
            fn += 1
        elif not exp and got:
            fp += 1
        else:
            tn += 1
    return {**_prf1(tp, fp, fn), "support_positive": tp + fn, "n": len(rows)}


def summarize_component_suite(
    rows: list[dict[str, Any]], *, correct_key: str = "correct"
) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"accuracy": 0.0, "n": 0}
    ok = sum(1 for r in rows if r.get(correct_key))
    return {"accuracy": ok / n, "n": n, "correct": ok}


def summarize_schema_suite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"aligned": {"accuracy": 0.0, "n": 0}, "strict": {"accuracy": 0.0, "n": 0}}
    strict_ok = sum(1 for r in rows if r.get("correct_strict", r.get("correct")))
    aligned_ok = sum(1 for r in rows if r.get("correct_aligned"))
    ind_ok = sum(1 for r in rows if r.get("indicator_ok"))
    val_ok = sum(1 for r in rows if r.get("value_ok"))
    time_ok = sum(1 for r in rows if r.get("time_ok"))
    mean_field = sum(r.get("field_score") or 0.0 for r in rows) / n
    return {
        "n": n,
        "accuracy": strict_ok / n,
        "correct": strict_ok,
        "accuracy_basis": "strict",
        "strict": {
            "accuracy": strict_ok / n,
            "correct": strict_ok,
            "description": "indicator + value + time all pass",
        },
        "aligned": {
            "accuracy": aligned_ok / n,
            "correct": aligned_ok,
            "description": "at least 2 of 3 fields pass",
            "mean_field_score": mean_field,
            "indicator_rate": ind_ok / n,
            "value_rate": val_ok / n,
            "time_rate": time_ok / n,
        },
    }


def summarize_verdict_suite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"aligned": {"accuracy": 0.0, "n": 0}, "strict": {"accuracy": 0.0, "n": 0}}
    strict_ok = sum(1 for r in rows if r.get("correct_strict", r.get("correct")))
    aligned_ok = sum(1 for r in rows if r.get("correct_aligned", r.get("correct")))
    if any(r.get("correct_aligned") is not None for r in rows):
        aligned_ok = sum(1 for r in rows if r.get("correct_aligned"))
    return {
        "n": n,
        "accuracy": strict_ok / n,
        "correct": strict_ok,
        "accuracy_basis": "strict",
        "strict": {
            "accuracy": strict_ok / n,
            "correct": strict_ok,
            "description": "exact verdict match vs verify_claim",
        },
        "aligned": {
            "accuracy": aligned_ok / n,
            "correct": aligned_ok,
            "description": "strict match, or mismatch + unverifiable in 10-30% gray zone",
        },
    }


def build_report(
    *,
    run_id: str,
    dataset_id: str,
    outcome: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "outcome": outcome or {},
        "audit": audit or {},
        "components": components or {},
    }
