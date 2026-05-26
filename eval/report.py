"""
eval.report — 결과 리스트 → 사람-읽기 좋은 리포트 (markdown + console).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_summary(agg: dict, rows: list[dict]) -> str:
    """전체 요약 마크다운."""
    lines = []
    lines.append("# StructVerify Evaluation Report\n")

    lines.append("## Summary (FEVER-style)\n")
    lines.append(f"- Total claims: **{agg['total_claims']}**")
    lines.append(f"- **Label Accuracy**: {agg.get('label_accuracy', agg.get('verdict_accuracy', 0)):.1%}")
    lines.append(f"- **FEVER Score** (strict): {agg.get('fever_score', 0):.1%}")
    lines.append(f"- **Macro F1**: {agg.get('macro_f1', 0):.3f}")
    lines.append(f"- Mismatch precision: {agg['mismatch_precision']:.1%}")
    lines.append(f"- Mismatch recall: {agg['mismatch_recall']:.1%}")
    if agg.get("avg_elapsed_sec") is not None:
        lines.append(f"- Avg elapsed/claim: {agg['avg_elapsed_sec']:.1f}s "
                     f"(n={agg['n_with_elapsed']})")
    lines.append("")

    if agg.get("per_class"):
        lines.append("## Per-class metrics (FEVER-style)\n")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("| --- | --- | --- | --- | --- |")
        for cls, m in agg["per_class"].items():
            lines.append(
                f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {m['support']} |"
            )
        lines.append("")

    lines.append("## Per-stage Accuracy\n")
    lines.append(f"| Stage | Accuracy |")
    lines.append(f"| --- | --- |")
    lines.append(f"| schema.indicator (부분 매칭) | {agg['indicator_partial_accuracy']:.1%} |")
    lines.append(f"| schema.value | {agg['schema_value_accuracy']:.1%} |")
    lines.append(f"| schema.time_period | {agg['schema_time_accuracy']:.1%} |")
    lines.append(f"| schema.population | {agg['schema_pop_accuracy']:.1%} |")
    lines.append(f"| evidence.stat_id | {agg['stat_id_accuracy']:.1%} |")
    lines.append(f"| evidence.official_value | {agg['value_accuracy']:.1%} |")
    lines.append("")

    lines.append("## Confusion Matrix (gold → actual)\n")
    lines.append("| gold → actual | count |")
    lines.append("| --- | --- |")
    for k, v in agg["confusion_matrix"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    if agg["failure_modes"]:
        lines.append("## Failure Modes\n")
        lines.append("| Mode | Count |")
        lines.append("| --- | --- |")
        for m, n in sorted(agg["failure_modes"].items(), key=lambda x: -x[1]):
            lines.append(f"| {m} | {n} |")
        lines.append("")

    # bad cases (incorrect verdict)
    bad = [r for r in rows if not r.get("verdict_correct")]
    if bad:
        lines.append(f"## Incorrect Verdict ({len(bad)}건)\n")
        lines.append("| claim | gold | actual | failure_mode | gold_value | actual_value |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in bad[:30]:
            txt = (r.get("claim_text") or "")[:50].replace("|", "/")
            lines.append(
                f"| {txt}... | {r.get('gold_verdict')} | {r.get('actual_verdict')} | "
                f"{r.get('failure_mode')} | {r.get('gold_value')} | {r.get('actual_value')} |"
            )
        if len(bad) > 30:
            lines.append(f"\n... +{len(bad)-30} more")
        lines.append("")

    return "\n".join(lines)


def write_reports(out_dir: Path, rows: list[dict], agg: dict) -> None:
    """results dir에 *_full.json, *_summary.md, *_rows.json 작성."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(
        render_summary(agg, rows), encoding="utf-8",
    )
    (out_dir / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def compare_to_baseline(rows: list[dict], baseline_rows: list[dict]) -> dict:
    """이전 baseline 대비 신규 win / 신규 regression 식별.

    Returns:
        {"wins": [...], "regressions": [...]}
    """
    base_by_claim = {r["claim_id"]: r for r in baseline_rows if r.get("claim_id")}
    wins, regs = [], []
    for r in rows:
        cid = r.get("claim_id")
        b = base_by_claim.get(cid)
        if b is None:
            continue
        if not b["verdict_correct"] and r["verdict_correct"]:
            wins.append({"claim_id": cid, "claim_text": r["claim_text"],
                         "baseline_verdict": b["actual_verdict"],
                         "now_verdict": r["actual_verdict"]})
        elif b["verdict_correct"] and not r["verdict_correct"]:
            regs.append({"claim_id": cid, "claim_text": r["claim_text"],
                         "baseline_verdict": b["actual_verdict"],
                         "now_verdict": r["actual_verdict"]})
    return {"wins": wins, "regressions": regs}
