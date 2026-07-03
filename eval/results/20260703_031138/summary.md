# StructVerify Evaluation Report

## Summary (FEVER-style) — mode: `oracle` (Setting A — gold claim_text 직접 입력 (detection 우회))

- Total claims: **120**
- **Label Accuracy**: 20.0%
- **FEVER Score** (strict): 18.3%
- **Macro F1**: 0.143
- Mismatch precision: 20.0%
- Mismatch recall: 4.7%
- Avg elapsed/claim: 154.0s (n=120)

## Per-class metrics (FEVER-style)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| match | 0.500 | 0.019 | 0.036 | 53 |
| mismatch | 0.200 | 0.047 | 0.075 | 43 |
| unverifiable | 0.194 | 0.875 | 0.318 | 24 |

## Per-stage Accuracy

| Stage | Accuracy |
| --- | --- |
| schema.indicator (substring) | 30.8% |
| schema.value | 75.0% |
| schema.time_period | 75.0% |
| schema.population | 45.0% |
| evidence.stat_id | 0.8% |
| evidence.official_value | 17.5% |

## Confusion Matrix (gold → actual)

| gold → actual | count |
| --- | --- |
| match→match | 1 |
| match→mismatch | 6 |
| match→unverifiable | 46 |
| mismatch→mismatch | 2 |
| mismatch→unverifiable | 41 |
| unverifiable→match | 1 |
| unverifiable→mismatch | 2 |
| unverifiable→unverifiable | 21 |

## Failure Modes

| Mode | Count |
| --- | --- |
| no_table_picked | 86 |
| wrong_table | 7 |
| wrong_value | 3 |

## Incorrect Verdict (96건)

| claim | gold | actual | failure_mode | gold_value | actual_value |
| --- | --- | --- | --- | --- | --- |
| 1991년 한국의 수상운송업 수익은 52백만 원을 기록하였다.... | match | unverifiable | no_table_picked | 52.0 | None |
| 1996년 한국의 운송 관련 서비스업 업체 수는 7개였다.... | match | unverifiable | no_table_picked | 7.0 | None |
| 1991 산업 및 지역별 총괄-수상운송업은(는) 59,8백만원로 나타났다.... | mismatch | unverifiable | no_table_picked | 52.0 | None |
| 1993년에 10,000kg 이하의 차량으로 등록된 수는 66,344대였습니다.... | mismatch | unverifiable | no_table_picked | 66344.0 | None |
| 1994 10,000Kg 이하은(는) 66,344대로 나타났다.... | match | unverifiable | no_table_picked | 66344.0 | None |
| 1994 10,000Kg 이하은(는) 82,930대로 나타났다.... | mismatch | unverifiable | no_table_picked | 66344.0 | None |
| 2007년 한국에서 총 13,635,462,684천 원이 연령별 진료비로 지출되었다.... | match | mismatch | wrong_table | 13635462684.0 | 1888371414.0 |
| 2007년 한국에서 총 17,726,101,489,2천 원이 질병 소분류별 입원 다빈도 상... | mismatch | unverifiable | no_table_picked | 13635462684.0 | None |
| 2007년 한국에서 총 14,790명이 질병 소분류별 입원 다빈도 상병 급여 현황으로 입원... | match | unverifiable | no_table_picked | 14790.0 | None |
| 1980년에 한국의 지식 재산권 처리 건수는 35,966건이었다.... | match | unverifiable | no_table_picked | 35966.0 | None |
| 1996년에 한국의 지식 재산권 처리 건수는 193건이었다.... | match | unverifiable | no_table_picked | 193.0 | None |
| 2024년 전국의 비주거용 부동산 임대 실적은 약 71,629개 ㎡로 집계되었으며, 이는 ... | match | unverifiable | no_table_picked | 71629.0 | None |
| 전국적으로 2005년에 향후 주택 구입을 희망하는 사람들이 약 400건으로 나타났습니다.... | match | unverifiable | no_table_picked | 400.0 | None |
| 전국적으로 2004년에 향후 주택 구입을 희망하는 사람들이 약 400건으로 나타났습니다.... | mismatch | unverifiable | no_table_picked | 400.0 | None |
| 2024년 2월 한국의 등표 관측 온도는 평균 3.2°C였습니다.... | match | unverifiable | no_table_picked | 3.2 | None |
| 2024년 2월 한국의 등표 관측 온도는 평균 3.68°C였습니다.... | mismatch | unverifiable | no_table_picked | 3.2 | None |
| 2024년 한국의 항공 기후 정보는 1022.3으로 측정되었습니다.... | match | unverifiable | no_table_picked | 1022.3 | None |
| 2024년 기준으로 방문 일 당 약제비의 크기에 따른 진료비 심사실적은 일일 평균 1.0으... | match | unverifiable | no_table_picked | 1.0 | None |
| 2018년에 65세 이상 노인들의 주요 질병에서 다빈도로 나타나는 상병들은 총 1,339,... | match | unverifiable | no_table_picked | 1339835.0 | None |
| 2025년에 기상산업 육성을 위해 우선 추진해야 할 정부 중점 정책적 지원 사항은 5개입니... | mismatch | unverifiable | no_table_picked | 5.0 | None |
| 2024년에도 기상산업 육성을 위해 우선 추진해야 할 정부 중점 정책적 지원 사항은 5개였... | match | unverifiable | no_table_picked | 5.0 | None |
| 2024년 기준 한국의 기상산업 사업체는 총 135개입니다.... | match | mismatch | wrong_table | 135.0 | 36117.0 |
| 2025년 한국의 인구는 약 7400만 명으로 추정됩니다.... | unverifiable | mismatch | wrong_value | None | 51117378.0 |
| 인구 증가 추세가 지속될 것으로 예상됩니다.... | unverifiable | mismatch | wrong_value | None | 52732700.0 |
| 2024년 전국 농어업인 가구의 10.125%가 악취를 주요 문제로 인식하고 있습니다.... | mismatch | unverifiable | no_table_picked | 13.5 | None |
| 2015년에는 전국 농어업인 마을에서 3,920명이 마을 안전 시설/설비에 대해 긍정적인 ... | match | unverifiable | no_table_picked | 3920.0 | None |
| 2024년 전국 농어업인 가구의 15.525%가 악취를 주요 문제로 인식하고 있습니다.... | mismatch | unverifiable | no_table_picked | 13.5 | None |
| 2011년에 최근 1년간 술집 출입을 시도한 아동 및 청소년은 총 328명이었습니다.... | match | unverifiable | no_table_picked | 328.0 | None |
| 2010년에 최근 1년간 술집 출입을 시도한 아동 및 청소년은 총 328명이었습니다.... | mismatch | unverifiable | no_table_picked | 328.0 | None |
| 2012년에 최근 1년간 술집 출입을 시도한 아동 및 청소년은 총 328명이었습니다.... | mismatch | unverifiable | no_table_picked | 328.0 | None |

... +66 more
