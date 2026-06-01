"""Aggregate 3-axis eval metrics into a single report."""
from __future__ import annotations

from collections import defaultdict
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
    """Aggregate outcome metrics for one schema mode."""
    n = len(predictions)
    if n == 0:
        return {
            "verdict_accuracy": 0.0,
            "value_tolerance_rate": 0.0,
            "n": 0,
        }
    verdict_ok = sum(1 for p in predictions if p.get("verdict_correct"))
    value_checked = [
        p for p in predictions if p.get("value_within_tolerance") is not None
    ]
    value_ok = sum(1 for p in value_checked if p.get("value_within_tolerance"))
    stat_checked = [
        p for p in predictions if p.get("stat_id_match") is not None
    ]
    stat_ok = sum(1 for p in stat_checked if p.get("stat_id_match"))
    vovw = sum(1 for p in predictions if p.get("value_ok_verdict_wrong"))
    errors = sum(1 for p in predictions if p.get("error"))

    by_expected: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "verdict_correct": 0, "value_within_tolerance": 0}
    )
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        exp = p.get("expected_verdict") or "unknown"
        pred = p.get("predicted_verdict") or "error"
        by_expected[exp]["n"] += 1
        if p.get("verdict_correct"):
            by_expected[exp]["verdict_correct"] += 1
        if p.get("value_within_tolerance"):
            by_expected[exp]["value_within_tolerance"] += 1
        confusion[exp][pred] += 1

    by_expected_summary = {}
    for exp, stats in by_expected.items():
        cnt = stats["n"]
        by_expected_summary[exp] = {
            "n": cnt,
            "verdict_accuracy": stats["verdict_correct"] / cnt if cnt else 0.0,
            "value_tolerance_rate": (
                stats["value_within_tolerance"] / cnt if cnt else 0.0
            ),
        }

    return {
        "n": n,
        "verdict_accuracy": verdict_ok / n,
        "value_tolerance_rate": (
            value_ok / len(value_checked) if value_checked else None
        ),
        "stat_id_match_rate": (
            stat_ok / len(stat_checked) if stat_checked else None
        ),
        "value_ok_verdict_wrong_rate": vovw / n,
        "value_ok_verdict_wrong": vovw,
        "errors": errors,
        "verdict_correct": verdict_ok,
        "by_expected_verdict": by_expected_summary,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def summarize_outcome_nested(
    predictions: list[dict[str, Any]],
    *,
    primary_schema_mode: str = "oracle",
) -> dict[str, Any]:
    """Nested oracle / induce summaries plus primary_schema_mode."""
    modes = sorted({p.get("schema_mode", "induce") for p in predictions})
    out: dict[str, Any] = {"primary_schema_mode": primary_schema_mode}
    for mode in modes:
        subset = [p for p in predictions if p.get("schema_mode") == mode]
        out[mode] = summarize_outcome(subset)
    # Flat primary metrics for backward-compatible readers
    primary_block = out.get(primary_schema_mode)
    if isinstance(primary_block, dict):
        out["verdict_accuracy"] = primary_block.get("verdict_accuracy")
        out["value_tolerance_rate"] = primary_block.get("value_tolerance_rate")
        out["n"] = primary_block.get("n")
    return out


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
    split: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "outcome": outcome or {},
        "audit": audit or {},
        "components": components or {},
    }
    if split:
        report["split"] = split
    return report
