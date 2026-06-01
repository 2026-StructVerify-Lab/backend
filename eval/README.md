# StructVerify Evaluation (3-axis)

Greenfield eval — no legacy E2E article harness.

## v1 vs v2

| Dataset | Role |
|---------|------|
| `structverify_outcome_v1` | **Frozen** smoke / historical comparison (do not rebuild in place) |
| `structverify_outcome_v2` | **Primary** KOSIS rebuild: natural claim text, validator, holdout split |

Harness default (`config/eval_run.yaml`): v2 dataset, **dual schema modes**.

## Dual KPI (outcome)

Each case runs twice:

| Mode | Schema | What it measures |
|------|--------|------------------|
| **oracle** (primary) | Gold `ClaimSchema` injected; `induce_schemas` skipped | Agent + KOSIS retrieval/verdict (aligned with v4 when schema is known) |
| **induce** (secondary) | Full `induce_schemas` E2E | Schema induction + agent (regression on schema path) |

`report.json` → `outcome.oracle` / `outcome.induce` with shared diagnostics:

- `verdict_accuracy`, `value_tolerance_rate`
- `stat_id_match_rate`, `value_ok_verdict_wrong_rate` (KOSIS OK but verdict wrong)
- `by_expected_verdict`, `confusion`

Tune on **train split only**; run **holdout** manually before release.

## Axes

| Axis | Purpose | Data |
|------|---------|------|
| **Outcome** | Task success: verdict vs gold | `eval/datasets/<id>/claims.jsonl` |
| **Audit** | Constraints + KOSIS ID grounding | Runs on **primary-mode** predictions only |
| **Components** | Per-layer regression | `eval/datasets/structverify_components_v2/*.jsonl` |

Do **not** tune agent parameters on component scores (diagnostic only).

## Build outcome dataset v2 (KOSIS-first)

```bash
python scripts/build_outcome_dataset.py --config config/eval_outcome_builder_v2.yaml
```

Produces:

- `eval/datasets/structverify_outcome_v2/claims.jsonl` + `manifest.json` (`holdout_case_ids`, `schema_version: 2`)
- `eval/datasets/structverify_components_v2/` (if `emit_component_fixtures: true`)

Requires KOSIS API key in env / `config/default.yaml`.

Refresh mismatch perturbation + component fixtures (no full KOSIS rebuild):

```bash
python scripts/refresh_eval_fixtures.py --config config/eval_outcome_builder_v2.yaml
```

## Run eval

```bash
# Train split (default), dual oracle+induce
python scripts/run_eval.py --axis outcome --limit 5

python scripts/run_eval.py --axis all --split train

# Holdout check (release gate, not for tuning)
python scripts/run_eval.py --axis outcome --split holdout

# v1 frozen comparison
python scripts/run_eval.py --dataset structverify_outcome_v1 --axis outcome
```

Reports: `eval/runs/<run_id>/report.json`

### One-page summary image

```bash
pip install matplotlib
python scripts/eval_report_card.py
python scripts/eval_report_card.py --report eval/runs/eval_structverify_outcome_v2_.../report.json
```

Dual outcome bars: oracle vs induce verdict (when nested outcome present).

## Config

- Builder v2: `config/eval_outcome_builder_v2.yaml`
- Harness: `config/eval_run.yaml` (merged with `config/default.yaml`)

Key harness fields:

```yaml
eval:
  schema_modes: [oracle, induce]
  primary_schema_mode: oracle
  workspace_scope: per_case   # external_job_id = case_id per case
  split: train
```

## Success signals (v2 train, oracle)

- `value_ok_verdict_wrong_rate` **< 5%**
- `verdict_accuracy` **> v1 induce-style baseline** (target 60%+)
- Components retrieval **≥ 90%**

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

Outcome harness skips `detect_claims`, injects oracle claim, runs via `structverify/eval/outcome/runtime_slice.py` (same `RuntimeAgent._verify_with_agent` path as production).
