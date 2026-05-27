# StructVerify Evaluation Report

## Summary (FEVER-style)

- Total claims: **2**
- **Label Accuracy**: 0.0%
- **FEVER Score** (strict): 0.0%
- **Macro F1**: 0.000
- Mismatch precision: 0.0%
- Mismatch recall: 0.0%
- Avg elapsed/claim: 104.7s (n=2)

## Per-class metrics (FEVER-style)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| match | 0.000 | 0.000 | 0.000 | 2 |
| mismatch | 0.000 | 0.000 | 0.000 | 0 |
| unverifiable | 0.000 | 0.000 | 0.000 | 0 |

## Per-stage Accuracy

| Stage | Accuracy |
| --- | --- |
| schema.indicator (substring) | 50.0% |
| schema.indicator (semantic τ=0.65) | 100.0% (avg sim=0.853, n=2) |
| schema.value | 100.0% |
| schema.time_period | 100.0% |
| schema.population | 50.0% |
| evidence.stat_id | 0.0% |
| evidence.official_value | 0.0% |

## Confusion Matrix (gold → actual)

| gold → actual | count |
| --- | --- |
| match→mismatch | 1 |
| match→unverifiable | 1 |

## Failure Modes

| Mode | Count |
| --- | --- |
| wrong_table | 1 |
| no_table_picked | 1 |

## Incorrect Verdict (2건)

| claim | gold | actual | failure_mode | gold_value | actual_value |
| --- | --- | --- | --- | --- | --- |
| 2022년 한국에서 경제적 이유로 인한 미충족 의료율은 243명으로 나타났다.... | match | mismatch | wrong_table | 243.0 | 1141.0 |
| 2022년 한국에서 입원 시 과잉 진료 경험이 5건으로 보고되었다.... | match | unverifiable | no_table_picked | 5.0 | None |
