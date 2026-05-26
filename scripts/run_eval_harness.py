#!/usr/bin/env python3
"""CLI — run EvalHarness on a frozen golden dataset."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structverify.eval.harness.runner import (
    EvalHarness,
    load_eval_harness_config,
    merge_harness_config,
)
from structverify.eval.harness.visualize import render_eval_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E2E eval harness on golden set")
    parser.add_argument(
        "--config",
        default="config/eval_harness.yaml",
        help="Harness YAML config",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset_id from config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to evaluate (default: all)",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Output runs directory",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id (subdir under runs_dir)",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Existing run directory (with --viz-only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume predictions.jsonl in --run or new run dir",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip chart/HTML generation",
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Regenerate charts/HTML from existing report.json in --run",
    )
    args = parser.parse_args()

    if args.viz_only:
        if not args.run:
            print("--viz-only requires --run PATH", file=sys.stderr)
            return 1
        ok = render_eval_report(args.run)
        print(f"Visualization {'done' if ok else 'skipped (install matplotlib)'}: {args.run}")
        return 0

    harness_cfg = load_eval_harness_config(args.config)
    merged = merge_harness_config(harness_cfg)
    dataset_id = args.dataset or harness_cfg.get("dataset_id", "structverify_eval_v4")
    runs_dir = Path(args.runs_dir or harness_cfg.get("runs_dir", "eval/runs"))
    datasets_root = Path(harness_cfg.get("datasets_root", "eval/datasets"))

    harness = EvalHarness(
        merged,
        datasets_root=datasets_root,
        dataset_id=dataset_id,
        runs_dir=runs_dir,
    )

    existing = args.run
    if args.resume and not existing and args.run_id:
        existing = runs_dir / args.run_id

    report = asyncio.run(
        harness.run(
            limit=args.limit,
            run_id=args.run_id,
            skip_viz=args.skip_viz,
            resume=args.resume,
            existing_run_dir=existing,
        )
    )
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
