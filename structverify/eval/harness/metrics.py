"""Aggregate harness metrics from per-article prediction records."""
from __future__ import annotations

from typing import Any

from structverify.eval.harness.matcher import VERDICT_LABELS

LABEL_TO_IDX = {v: i for i, v in enumerate(VERDICT_LABELS)}


def _prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def compute_report(
    dataset_id: str,
    run_id: str,
    article_records: list[dict[str, Any]],
    *,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    claims_gold = 0
    matched_total = 0
    verdict_correct_total = 0
    stat_id_checked = 0
    stat_id_correct_total = 0
    confusion = [[0 for _ in VERDICT_LABELS] for _ in VERDICT_LABELS]
    per_domain: dict[str, dict[str, int]] = {}
    per_article: list[dict[str, Any]] = []
    errors = 0

    for rec in article_records:
        if rec.get("error"):
            errors += 1
        domain = rec.get("intended_domain", "unknown")
        if domain not in per_domain:
            per_domain[domain] = {
                "gold": 0,
                "matched": 0,
                "verdict_correct": 0,
            }
        g = rec.get("gold_claims", 0)
        m = rec.get("matched_claims", 0)
        c = rec.get("verdict_correct", 0)
        claims_gold += g
        matched_total += m
        verdict_correct_total += c
        per_domain[domain]["gold"] += g
        per_domain[domain]["matched"] += m
        per_domain[domain]["verdict_correct"] += c

        acc = (c / m) if m else 0.0
        per_article.append(
            {
                "article_id": rec.get("article_id"),
                "accuracy": acc,
                "matched": m,
                "gold": g,
                "error": rec.get("error"),
            }
        )

        for cl in rec.get("claims", []):
            gold_v = cl.get("gold_verdict")
            pred_v = cl.get("predicted_verdict")
            if cl.get("matched") and gold_v in LABEL_TO_IDX and pred_v in LABEL_TO_IDX:
                confusion[LABEL_TO_IDX[gold_v]][LABEL_TO_IDX[pred_v]] += 1
            if cl.get("stat_id_correct") is not None:
                stat_id_checked += 1
                if cl.get("stat_id_correct"):
                    stat_id_correct_total += 1

    per_class_tp_fp_fn: dict[str, dict[str, int]] = {
        v: {"tp": 0, "fp": 0, "fn": 0} for v in VERDICT_LABELS
    }
    for gi, gold_label in enumerate(VERDICT_LABELS):
        for pi, pred_label in enumerate(VERDICT_LABELS):
            n = confusion[gi][pi]
            if gold_label == pred_label:
                per_class_tp_fp_fn[gold_label]["tp"] += n
            else:
                per_class_tp_fp_fn[gold_label]["fn"] += n
                per_class_tp_fp_fn[pred_label]["fp"] += n

    per_verdict = {
        label: _prf1(v["tp"], v["fp"], v["fn"])
        for label, v in per_class_tp_fp_fn.items()
    }

    per_domain_out = {
        d: {
            "accuracy": (
                stats["verdict_correct"] / stats["matched"]
                if stats["matched"]
                else 0.0
            ),
            "extraction_recall": (
                stats["matched"] / stats["gold"] if stats["gold"] else 0.0
            ),
            "matched": stats["matched"],
            "gold": stats["gold"],
        }
        for d, stats in sorted(per_domain.items())
    }

    verdict_accuracy = (
        verdict_correct_total / matched_total if matched_total else 0.0
    )
    extraction_recall = matched_total / claims_gold if claims_gold else 0.0
    stat_id_accuracy = (
        stat_id_correct_total / stat_id_checked if stat_id_checked else 0.0
    )

    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "articles_total": len(article_records),
        "articles_with_errors": errors,
        "claims_gold": claims_gold,
        "claims_matched": matched_total,
        "manifest_articles_sha256": manifest_sha256,
        "summary": {
            "verdict_accuracy": verdict_accuracy,
            "extraction_recall": extraction_recall,
            "stat_id_accuracy": stat_id_accuracy,
        },
        "per_verdict": per_verdict,
        "confusion_matrix": {
            "labels": list(VERDICT_LABELS),
            "matrix": confusion,
        },
        "per_domain": per_domain_out,
        "per_article": per_article,
    }
