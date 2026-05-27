# StructVerify Evaluation Report

## Summary (FEVER-style)

- Total claims: **5**
- **Label Accuracy**: 20.0%
- **FEVER Score** (strict): 20.0%
- **Macro F1**: 0.111
- Mismatch precision: 0.0%
- Mismatch recall: 0.0%
- Avg elapsed/claim: 144.6s (n=5)

## Per-class metrics (FEVER-style)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| match | 0.000 | 0.000 | 0.000 | 2 |
| mismatch | 0.000 | 0.000 | 0.000 | 1 |
| unverifiable | 0.250 | 0.500 | 0.333 | 2 |

## Per-stage Accuracy

| Stage | Accuracy |
| --- | --- |
| schema.indicator (부분 매칭) | 0.0% |
| schema.value | 60.0% |
| schema.time_period | 80.0% |
| schema.population | 40.0% |
| evidence.stat_id | 0.0% |
| evidence.official_value | 20.0% |

## Confusion Matrix (gold → actual)

| gold → actual | count |
| --- | --- |
| match→unverifiable | 2 |
| mismatch→unverifiable | 1 |
| unverifiable→mismatch | 1 |
| unverifiable→unverifiable | 1 |

## Failure Modes

| Mode | Count |
| --- | --- |
| no_table_picked | 3 |
| wrong_value | 1 |

## Incorrect Verdict (4건)

| claim | gold | actual | failure_mode | gold_value | actual_value |
| --- | --- | --- | --- | --- | --- |
| 1991년 한국의 수상운송업 수익은 52백만 원을 기록하였다.... | match | unverifiable | no_table_picked | 52.0 | None |
| 1996년 한국의 운송 관련 서비스업 업체 수는 7개였다.... | match | unverifiable | no_table_picked | 7.0 | None |
| 1991 산업 및 지역별 총괄-수상운송업은(는) 59,8백만원로 나타났다.... | mismatch | unverifiable | no_table_picked | 52.0 | None |
| 최근 한국의 인구 규모는 약 217만 명 정도로 파악되고 있다.... | unverifiable | mismatch | wrong_value | None | 51751065.0 |
