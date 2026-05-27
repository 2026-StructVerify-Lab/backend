# StructVerify Evaluation (3-axis)

Greenfield eval — no legacy E2E article harness.

## Axes

| Axis | Purpose | Data |
|------|---------|------|
| **Outcome** | Task success: verdict vs gold (value-based, not stat_id) | `eval/datasets/<id>/claims.jsonl` |
| **Audit** | Constraints + KOSIS ID grounding | No labels; runs on predictions |
| **Components** | Per-layer regression | `eval/datasets/structverify_components_v1/*.jsonl` |

## Build outcome dataset (KOSIS-first)

```bash
python scripts/build_outcome_dataset.py --config config/eval_outcome_builder.yaml
```

Produces:

- `eval/datasets/structverify_outcome_v1/claims.jsonl`
- `eval/datasets/structverify_components_v1/` (if `emit_component_fixtures: true`)

Requires KOSIS API key in env / `config/default.yaml`.

After changing perturbation or fixture rules (no full KOSIS rebuild):

```bash
python scripts/refresh_eval_fixtures.py --config config/eval_outcome_builder.yaml
```

## Run eval

```bash
# Outcome only
python scripts/run_eval.py --axis outcome --limit 5

# Outcome + audit + components
python scripts/run_eval.py --axis all --limit 5

# Components only (verdict suite uses mock evidence — no KOSIS)
python scripts/run_eval.py --axis components --suite verdict

# Audit on existing predictions
python scripts/run_eval.py --axis audit --predictions eval/runs/.../predictions.jsonl
```

Reports: `eval/runs/<run_id>/report.json`

### One-page summary image

```bash
pip install matplotlib   # or: pip install -e ".[eval-viz]"
python scripts/eval_report_card.py
python scripts/eval_report_card.py --report eval/runs/eval_structverify_outcome_v1_.../report.json
```

Writes `report_card.png` beside `report.json` (horizontal bars, Outcome/Audit vs Components).

### Component metrics (report.json)

| Suite | `aligned` | `strict` |
|-------|-----------|----------|
| **schema** `accuracy` (top-level) | — | **strict**: indicator + value + time all pass |
| **schema** `aligned` | ≥2/3 fields (diagnostic) | — |
| **verdict** `accuracy` (top-level) | — | **strict**: exact match vs `verify_claim` |
| **verdict** `aligned` | gray-zone `unverifiable` counts for mismatch (diagnostic) | — |

Top-level `accuracy` equals `strict.accuracy` (`accuracy_basis: "strict"`). There is no third duplicate score.

## Config

- Builder: `config/eval_outcome_builder.yaml`
- Harness: `config/eval_run.yaml` (merged with `config/default.yaml`)

## Outcome case schema (one JSONL line)

```json
{
  "case_id": "outcome_healthcare_0001",
  "case_type": "atomic",
  "claim_text": "2022년 고용률은 62.3%로 집계됐다.",
  "expected_verdict": "match",
  "indicator": "고용률",
  "time_period": "2022",
  "unit": "%",
  "stated_value": 62.3,
  "official_value": 62.3,
  "domain": "employment",
  "reference_stat_id": "DT_200Y108",
  "kosis_org_id": "200",
  "label_method": "kosis_probe"
}
```

`kosis_org_id` is the KOSIS `orgId` (from catalog or parsed from `DT_{orgId}_{tblId}`) for manual lookup on [KOSIS](https://kosis.kr/).

Outcome harness skips `detect_claims`, injects oracle claim, runs schema + verify via `structverify/eval/outcome/runtime_slice.py`.
