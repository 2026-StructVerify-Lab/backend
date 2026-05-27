#!/usr/bin/env python3
"""Generate a one-page PNG summary from eval report.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structverify.eval.visualize_report import (
    find_latest_report,
    load_report,
    render_report_card,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval report → summary PNG")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to report.json (default: latest under eval/runs)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: next to report as report_card.png)",
    )
    p.add_argument("--runs-root", type=Path, default=Path("eval/runs"))
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report
    if report_path is None:
        report_path = find_latest_report(args.runs_root)
        if report_path is None:
            raise SystemExit(f"No report.json under {args.runs_root}")
    report_path = report_path.resolve()
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")

    out = args.output
    if out is None:
        out = report_path.parent / "report_card.png"
    out = out.resolve()

    report = load_report(report_path)
    render_report_card(report, out, dpi=args.dpi)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
