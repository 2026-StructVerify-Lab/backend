# StructVerify Evaluation Report

## Summary

- Total claims: **3**
- Verdict accuracy: **0.0%**
- Mismatch precision: 0.0%
- Mismatch recall: 0.0%

## Per-stage Accuracy

| Stage | Accuracy |
| --- | --- |
| schema.indicator (부분 매칭) | 0.0% |
| schema.value | 0.0% |
| schema.time_period | 0.0% |
| schema.population | 0.0% |
| evidence.stat_id | 0.0% |
| evidence.official_value | 0.0% |

## Confusion Matrix (gold → actual)

| gold → actual | count |
| --- | --- |
| match→exception | 2 |
| mismatch→exception | 1 |

## Failure Modes

| Mode | Count |
| --- | --- |
| exception | 3 |

## Incorrect Verdict (3건)

| claim | gold | actual | failure_mode | gold_value | actual_value |
| --- | --- | --- | --- | --- | --- |
| 1991년 한국의 수상운송업 수익은 52백만 원을 기록하였다.... | match | exception | exception | None | None |
| 1996년 한국의 운송 관련 서비스업 업체 수는 7개였다.... | match | exception | exception | None | None |
| 1991 산업 및 지역별 총괄-수상운송업은(는) 59,8백만원로 나타났다.... | mismatch | exception | exception | None | None |
