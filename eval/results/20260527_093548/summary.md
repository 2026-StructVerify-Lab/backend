# StructVerify Evaluation Report

## Summary (FEVER-style)

- Total claims: **1**
- **Label Accuracy**: 0.0%
- **FEVER Score** (strict): 0.0%
- **Macro F1**: 0.000
- Mismatch precision: 0.0%
- Mismatch recall: 0.0%
- Avg elapsed/claim: 237.8s (n=1)

## Per-class metrics (FEVER-style)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| match | 0.000 | 0.000 | 0.000 | 1 |
| mismatch | 0.000 | 0.000 | 0.000 | 0 |
| unverifiable | 0.000 | 0.000 | 0.000 | 0 |

## Per-stage Accuracy

| Stage | Accuracy |
| --- | --- |
| schema.indicator (substring) | 0.0% |
| schema.indicator (semantic τ=0.65) | 0.0% (avg sim=0.571, n=1) |
| schema.value | 100.0% |
| schema.time_period | 100.0% |
| schema.population | 0.0% |
| evidence.stat_id | 0.0% |
| evidence.official_value | 0.0% |

## Confusion Matrix (gold → actual)

| gold → actual | count |
| --- | --- |
| match→unverifiable | 1 |

## Failure Modes

| Mode | Count |
| --- | --- |
| no_table_picked | 1 |

## Incorrect Verdict (1건)

| claim | gold | actual | failure_mode | gold_value | actual_value |
| --- | --- | --- | --- | --- | --- |
| 1991년 한국의 수상운송업 수익은 52백만 원을 기록하였다.... | match | unverifiable | no_table_picked | 52.0 | None |
