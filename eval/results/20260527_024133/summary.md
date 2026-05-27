# StructVerify Evaluation Report

## Summary (FEVER-style)

- Total claims: **120**
- **Label Accuracy**: 20.0%
- **FEVER Score** (strict): 14.2%
- **Macro F1**: 0.175
- Mismatch precision: 28.6%
- Mismatch recall: 14.0%
- Avg elapsed/claim: 152.2s (n=120)

## Per-class metrics (FEVER-style)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| match | 0.667 | 0.038 | 0.071 | 53 |
| mismatch | 0.286 | 0.140 | 0.187 | 43 |
| unverifiable | 0.167 | 0.667 | 0.267 | 24 |

## Per-stage Accuracy

| Stage | Accuracy |
| --- | --- |
| schema.indicator (substring) | 30.8% |
| schema.indicator (semantic τ=0.65) | 71.4% (avg sim=0.733, n=112) |
| schema.value | 74.2% |
| schema.time_period | 80.0% |
| schema.population | 47.5% |
| evidence.stat_id | 1.7% |
| evidence.official_value | 13.3% |

## Confusion Matrix (gold → actual)

| gold → actual | count |
| --- | --- |
| match→match | 2 |
| match→mismatch | 7 |
| match→unverifiable | 44 |
| mismatch→match | 1 |
| mismatch→mismatch | 6 |
| mismatch→unverifiable | 36 |
| unverifiable→mismatch | 8 |
| unverifiable→unverifiable | 16 |

## Failure Modes

| Mode | Count |
| --- | --- |
| no_table_picked | 78 |
| wrong_value | 9 |
| wrong_table | 9 |

## Incorrect Verdict (96건)

| claim | gold | actual | failure_mode | gold_value | actual_value |
| --- | --- | --- | --- | --- | --- |
| 1991년 한국의 수상운송업 수익은 52백만 원을 기록하였다.... | match | unverifiable | no_table_picked | 52.0 | None |
| 1996년 한국의 운송 관련 서비스업 업체 수는 7개였다.... | match | unverifiable | no_table_picked | 7.0 | None |
| 1991 산업 및 지역별 총괄-수상운송업은(는) 59,8백만원로 나타났다.... | mismatch | unverifiable | no_table_picked | 52.0 | None |
| 최근 한국의 인구 규모는 약 217만 명 정도로 파악되고 있다.... | unverifiable | mismatch | wrong_value | None | 51751065.0 |
| 최근 한국의 인구 규모는 약 619만 명 정도로 파악되고 있다.... | unverifiable | mismatch | wrong_value | None | 5491666.0 |
| 1993년에 10,000kg 이하의 차량으로 등록된 수는 66,344대였습니다.... | mismatch | unverifiable | no_table_picked | 66344.0 | None |
| 1994 10,000Kg 이하은(는) 66,344대로 나타났다.... | match | unverifiable | no_table_picked | 66344.0 | None |
| 1994 10,000Kg 이하은(는) 82,930대로 나타났다.... | mismatch | unverifiable | no_table_picked | 66344.0 | None |
| 2007년 한국에서 총 13,635,462,684천 원이 연령별 진료비로 지출되었다.... | match | unverifiable | no_table_picked | 13635462684.0 | None |
| 2007년 한국에서 총 17,726,101,489,2천 원이 질병 소분류별 입원 다빈도 상... | mismatch | unverifiable | no_table_picked | 13635462684.0 | None |
| 2007년 한국에서 총 14,790명이 질병 소분류별 입원 다빈도 상병 급여 현황으로 입원... | match | unverifiable | no_table_picked | 14790.0 | None |
| 2024년 한국에서 전체 학생들 중 약 42.0%가 비공식 교육을 받고 있다.... | unverifiable | mismatch | wrong_value | None | 325511.0 |
| 한국의 경제 규모는 최근 약 930조 원에 달하는 것으로 나타났다.... | unverifiable | mismatch | wrong_value | None | 682814.6 |
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
| 2024년 기준 한국의 기상산업 사업체는 총 135개입니다.... | match | mismatch | wrong_table | 135.0 | 1134.0 |
| 2025년 한국의 인구는 약 7400만 명으로 추정됩니다.... | unverifiable | mismatch | wrong_value | None | 51684564.0 |
| 2024년 전국 농어업인 가구의 10.125%가 악취를 주요 문제로 인식하고 있습니다.... | mismatch | unverifiable | no_table_picked | 13.5 | None |
| 2015년에는 전국 농어업인 마을에서 3,920명이 마을 안전 시설/설비에 대해 긍정적인 ... | match | unverifiable | no_table_picked | 3920.0 | None |
| 2024년 전국 농어업인 가구의 15.525%가 악취를 주요 문제로 인식하고 있습니다.... | mismatch | unverifiable | wrong_table | 13.5 | 0.0 |
| 2011년에 최근 1년간 술집 출입을 시도한 아동 및 청소년은 총 328명이었습니다.... | match | unverifiable | no_table_picked | 328.0 | None |

... +66 more
