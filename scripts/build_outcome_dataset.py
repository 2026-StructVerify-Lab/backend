#!/usr/bin/env python3
"""Build outcome eval dataset (KOSIS-first, no synthetic articles)."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structverify.eval.build.outcome_builder import OutcomeDatasetBuilder
from structverify.eval.io import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build structverify outcome eval dataset")
    p.add_argument(
        "--config",
        default="config/eval_outcome_builder.yaml",
        help="Builder config YAML",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    builder = OutcomeDatasetBuilder(cfg)
    manifest = await builder.build()
    holdout = len(manifest.holdout_case_ids or [])
    print(
        f"Done: {manifest.dataset_id} cases={manifest.case_count} "
        f"match={manifest.match_count} mismatch={manifest.mismatch_count} "
        f"holdout={holdout} schema_version={manifest.schema_version} "
        f"status={manifest.status}"
    )


if __name__ == "__main__":
    asyncio.run(main())
