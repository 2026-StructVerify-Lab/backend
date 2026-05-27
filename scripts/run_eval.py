#!/usr/bin/env python3
"""Run 3-axis eval: outcome, audit, components."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structverify.eval.audit.runner import run_audit_on_predictions
from structverify.eval.components.runner import ComponentEvalRunner
from structverify.eval.io import load_yaml, read_jsonl, write_json
from structverify.eval.outcome.runner import OutcomeEvalRunner
from structverify.eval.report import build_report
from structverify.eval.io import make_run_id, ensure_run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StructVerify 3-axis eval")
    p.add_argument(
        "--axis",
        choices=["outcome", "audit", "components", "all"],
        default="all",
    )
    p.add_argument("--config", default="config/eval_run.yaml")
    p.add_argument("--dataset", default=None, help="Outcome dataset id override")
    p.add_argument("--suite", default=None, help="Component suite: detection|schema|retrieval|verdict")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--predictions",
        default=None,
        help="Predictions jsonl for audit-only run",
    )
    p.add_argument("--run-id", default=None)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    if args.dataset:
        cfg["dataset_id"] = args.dataset

    runs_root = Path(cfg.get("runs_root", "eval/runs"))
    outcome_report: dict = {}
    audit_summary: dict = {}
    components_summary: dict = {}
    run_id = args.run_id or make_run_id(
        cfg.get("dataset_id", "structverify_outcome_v1"), prefix="eval"
    )
    out_dir = ensure_run_dir(runs_root, run_id)

    if args.axis in ("outcome", "all"):
        runner = OutcomeEvalRunner(cfg)
        outcome_report = await runner.run(limit=args.limit, run_id=run_id)
        predictions = read_jsonl(Path(outcome_report["predictions_path"]))

        if args.axis == "all":
            cases = {c.case_id: c for c in runner.load_cases(limit=args.limit)}
            audit_rows, audit_summary = await run_audit_on_predictions(
                predictions, cases, runner.config
            )
            from structverify.eval.io import append_jsonl

            audit_path = out_dir / "audit.jsonl"
            if audit_path.exists():
                audit_path.unlink()
            for row in audit_rows:
                append_jsonl(audit_path, row)

    elif args.axis == "audit":
        pred_path = Path(args.predictions or "")
        if not pred_path.exists():
            raise SystemExit("--predictions required for audit axis")
        predictions = read_jsonl(pred_path)
        runner = OutcomeEvalRunner(cfg)
        cases = {c.case_id: c for c in runner.load_cases()}
        audit_rows, audit_summary = await run_audit_on_predictions(
            predictions, cases, runner.config
        )
        from structverify.eval.io import append_jsonl

        audit_path = out_dir / "audit.jsonl"
        if audit_path.exists():
            audit_path.unlink()
        for row in audit_rows:
            append_jsonl(audit_path, row)

    if args.axis in ("components", "all"):
        comp = ComponentEvalRunner(cfg)
        if args.suite:
            components_summary = {args.suite: await comp.run_suite(args.suite)}
        else:
            components_summary = await comp.run_all()

    full = build_report(
        run_id=run_id,
        dataset_id=cfg.get("dataset_id", ""),
        outcome=outcome_report.get("outcome") or outcome_report,
        audit=audit_summary,
        components=components_summary,
    )
    write_json(out_dir / "report.json", full)
    print(f"Report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
