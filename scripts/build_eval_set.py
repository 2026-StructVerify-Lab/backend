#!/usr/bin/env python3
"""CLI entry point for EvalSetBuilder."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# repo root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structverify.eval.builder import EvalSetBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KOSIS-first synthetic eval set")
    parser.add_argument(
        "--config",
        default="config/eval_builder.yaml",
        help="Path to eval builder YAML config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from eval/builder/.build_state.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print quota plan only (no generation)",
    )
    args = parser.parse_args()

    builder = EvalSetBuilder(args.config)
    manifest = asyncio.run(
        builder.run(dry_run=args.dry_run, resume=args.resume)
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
