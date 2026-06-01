#!/usr/bin/env python3
"""Re-apply value perturbation on mismatch claims and regenerate component fixtures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structverify.eval.build.claim_templates import (
    build_match_claim_text,
    perturb_stated_value,
)
from structverify.eval.build.claim_validator import validate_outcome_case
from structverify.eval.build.outcome_builder import _emit_component_fixtures
from structverify.eval.io import load_models, load_yaml, sha256_file, write_json
from structverify.eval.schemas import OutcomeCase, OutcomeManifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh outcome claims perturbation + component fixtures")
    p.add_argument("--config", default="config/eval_outcome_builder.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument(
        "--validate",
        action="store_true",
        help="Re-run claim_validator on all cases after refresh",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(ROOT / args.config)
    dataset_id = args.dataset or cfg["dataset_id"]
    datasets_root = Path(cfg.get("datasets_root", "eval/datasets"))
    claims_path = datasets_root / dataset_id / "claims.jsonl"

    cases = load_models(claims_path, OutcomeCase)
    updated = 0
    for case in cases:
        if case.expected_verdict != "mismatch" or case.label_method != "value_perturbation":
            continue
        if case.official_value is None:
            continue
        stated = perturb_stated_value(case.official_value, case.unit)
        case.stated_value = stated
        if case.indicator and case.time_period is not None:
            case.claim_text = build_match_claim_text(
                indicator=case.indicator,
                time_period=case.time_period,
                value=stated,
                unit=case.unit,
            )
        updated += 1

    with open(claims_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")

    match_n = sum(1 for c in cases if c.expected_verdict == "match")
    mismatch_n = sum(1 for c in cases if c.expected_verdict == "mismatch")
    manifest_path = datasets_root / dataset_id / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["claims_sha256"] = sha256_file(claims_path)
        data["match_count"] = match_n
        data["mismatch_count"] = mismatch_n
        write_json(manifest_path, data)

    comp_id = cfg.get("component_dataset_id", "structverify_components_v1")
    _emit_component_fixtures(cases, datasets_root, comp_id)

    if args.validate:
        bad = [
            (c.case_id, validate_outcome_case(c).errors)
            for c in cases
            if not validate_outcome_case(c).ok
        ]
        if bad:
            raise SystemExit(
                f"Validation failed for {len(bad)} case(s): {bad[:3]}..."
            )

    print(
        f"Updated {updated} mismatch claims in {claims_path}\n"
        f"Regenerated component fixtures under {datasets_root / comp_id}"
    )


if __name__ == "__main__":
    main()
