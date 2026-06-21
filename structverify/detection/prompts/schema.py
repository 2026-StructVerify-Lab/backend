"""detection/prompts/schema.py — Step 5 schema induction 프롬프트·JSON schema.

schema_inductor.py에서 분리 (문자열·상수 move-only, 동작 변경 없음).

[김예슬 - 2026-04-22] SCHEMA_INDUCTION_PROMPT, CLAIM_SCHEMA_JSON_SCHEMA
[v6.11 - 2026-05-12] parent_path / is_approximate / modifier
[박재윤 - 2026-05-14~18] 수치 추출·source_phrase 규칙 보강
[2026-05-27] REGENERATE_SCHEMA_PROMPT — replan tool용 schema 재분류
"""
from __future__ import annotations

from typing import Any


# ── 도메인별 indicator 힌트 (LLM 가이드) ────────────────────────────────
DOMAIN_HINTS: dict[str, str] = {
    "agriculture": "농가 수, 경작면적, 수확량, 고령화비율, 후계농 비율, 농업소득 등",
    "economy":     "경제성장률, 소비자물가지수, 수출액, 취업자 수, 산업생산지수 등",
    "finance":     "금리, 환율, 주가지수, 대출잔액, 가계부채비율 등",
    "population":  "인구수, 합계출산율, 기대수명, 고령화비율, 출생아 수 등",
    "employment":  "고용률, 실업률, 임금, 취업자 수, 근로시간, 쉬었음 인구 등",
    "healthcare":  "의료기관 수, 사망률, 질환자 수, 의료비, 건강보험료 등",
    "education":   "학생 수, 진학률, 교육비, 학교 수, 졸업률 등",
    "environment": "기온, 강수량, 적설량, 미세먼지 농도, 온실가스 배출량 등",
}


# ── JSON Schema (Structured Outputs 강제) ────────────────────────────────
# 박재유 ExtractResponse 스타일 + 우리 라이브러리 필드 통합
CLAIM_SCHEMA_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "indicator": {
            "type": "string",
            "description": (
                "측정 대상 *자체*만. (예: '쉬었음 인구', '출생아 수', '합계출산율') "
                "측정 행위 단어('증가율/변화/차이/상승/하락')는 indicator에서 빼세요. "
                "분류 축 단어('대졸이상/청년/수도권')는 population으로 분리하세요."
            ),
        },
        "time_period": {
            "type": "string",
            "description": (
                "기준 시점. 형식: 'YYYY' 또는 'YYYY-MM'. "
                "예: '2024', '2024-10'. 한글 표기('2024년 10월') 금지.\n"
                "★ 월 추론 규칙 — 원문 자연어 시점 표현을 *적극적으로* month까지 해석:\n"
                "  - '2025년 마지막 월/연말/12월 말'   → '2025-12'\n"
                "  - '2025년 초/연초/1월 초/신년'      → '2025-01'\n"
                "  - '2025년 상반기 말/6월 말'         → '2025-06'\n"
                "  - '2025년 하반기 초/7월 초'         → '2025-07'\n"
                "  - '2025년 1분기'                    → '2025-03' (분기 말월)\n"
                "  - '2025년 4분기/2025년 말'          → '2025-12'\n"
                "  - '2025년 봄'                       → '2025-04' (대표 월)\n"
                "  - '지난달' + anchor='2025-04'       → '2025-03'\n"
                "  - '최근/현재' + anchor_year=2025    → '2025' (월 단서 없음)\n"
                "★ 추론 가드 — 환각 금지:\n"
                "  - 원문에 *시점 표현이 전혀 없으면* anchor_year만 쓰거나 빈 문자열. 추론 금지.\n"
                "  - '월/분기/초/말/상반기/하반기/봄/여름/가을/겨울' 등 *명시적 단서가 있을 때만* month 추론.\n"
                "★ 추론 근거가 된 표현은 source_phrase 또는 *해당 schema의 다른 phrase 필드*에 원문 그대로 남깁니다."
            ),
        },
        "unit": {
            "type": "string",
            "description": (
                "수치 단위. 절대 비우지 말고 의미 있는 값을 넣으세요. "
                "명확한 단위(%/명/원/건/℃) 있으면 그대로. "
                "불분명한 경우: 기준점 대비 상대값=지수, 배수=배, 순위=위, 점수=점."
            ),
        },
        "population": {
            "type": "string",
            "description": (
                "대상 집단/범위. indicator에서 분리된 분류 축. "
                "(예: '대졸 이상 청년', '전국', '15~64세'). 없으면 '전체'."
            ),
        },
        "value": {
            "type": "number",
            "description": (
                "수치를 *기본 단위로 환산한* 순수 숫자. "
                "한글 단위 변환: '21만 7천명' → 217000, '3,200만 배럴' → 32000000. "
                "'34년 만에 최대' 같은 순위 표현의 N은 value로 쓰지 마세요."
            ),
        },
        "is_approximate": {
            "type": "boolean",
            "description": "근사 표현(안팎/이상/이하/약/가량) 있으면 true.",
        },
        "modifier": {
            "type": "string",
            "description": "근사 표현 원문 (예: '안팎', '이상'). 없으면 빈 문자열.",
        },
        "parent_path": {
            "type": "string",
            "description": (
                "KOSIS 카테고리 계층 '대분류 > 중분류 > 소분류'. "
                "예: '노동 > 청년 > 쉬었음 인구', '인구 > 출생 > 합계출산율'. "
                "기사 제목/출처/기관명 금지. "
                "KOSIS 대분류: 인구/가구/고용/노동/임금/물가/가계/보건/사회/복지/"
                "교육/환경/농림/수산/건설/주택/토지/교통/정보통신/경제/산업/무역."
            ),
        },
        "source_reference": {
            "type": "string",
            "description": "주장에 언급된 출처 기관/보고서 (없으면 빈 문자열).",
        },
        "source_phrase": {
            "type": "string",
            "description": (
                "★ 이 수치가 추출된 *검증 대상 문장의 원문 그대로의 문구*. "
                "예: '2만 171명', '6.7%', '0.76명', '1만 7921건'. "
                "이 문구는 *반드시* 검증 대상 문장에 그대로 등장해야 함. "
                "검증 대상 문장 밖(문맥)의 수치를 추출하는 경우 사용 금지."
            ),
        },
        "prev_value": {
            "type": ["number", "null"],
            "description": (
                "★ [v6.14 C2] 증가율/변화량 schema *전용 필드*. "
                "비교 기준값(전년/전월/이전 시점의 값)이 *검증 대상 문장*에 있으면 추출. "
                "예: '지난해 같은 달(1만 9059명)보다 6.7% 늘었다' → 6.7% schema의 prev_value=19059. "
                "예: '0.76명으로 지난해보다 0.04명 증가' → 0.04 schema의 prev_value=0.72 "
                "(0.76 - 0.04 = 0.72 직접 추출 또는 계산). "
                "절대값 schema(예: 6.7% schema가 아닌 20171 schema)에는 null 또는 생략. "
                "기준값이 문장에 없으면 null."
            ),
        },
        "prev_time_period": {
            "type": "string",
            "description": (
                "★ 비교 기준 시점. 형식: 'YYYY' 또는 'YYYY-MM'. "
                "★★ 중요: prev_value가 문장에 없어도(null이어도) "
                "비교 시점 표현만 있으면 *반드시* 채우세요. "
                "증가율/변화량 schema는 '무엇과 비교했는지'가 핵심이므로 "
                "시점은 본문 표현을 보고 계산: "
                "  현재 2023 + '1년 전/전년/지난해' → '2022'  "
                "  현재 2023 + '5년 전/5년 새'      → '2018'  "
                "  현재 2025-04 + '지난해 같은 달'  → '2024-04'  "
                "  현재 2024 + '2019년 대비'        → '2019'. "
                "즉 prev_value=null이어도 prev_time_period는 채웁니다 "
                "(나중에 통계 DB에서 그 시점 값을 직접 조회해 검증함). "
                "비교 시점 표현이 전혀 없으면만 빈 문자열."
            ),
        },
        "prev_phrase": {
            "type": "string",
            "description": (
                "prev_value가 추출된 *원문 문구* (source_phrase와 동일한 검증 규칙). "
                "예: '1만 9059명'. 검증 대상 문장에 *literally* 등장해야 함. "
                "prev_value가 *계산값*(예: 0.76-0.04=0.72)이면 빈 문자열."
            ),
        },
        "graph_schema_candidates": {
            "type": "array",
            "description": "Knowledge Graph 노드/엣지 후보",
            "items": {
                "type": "object",
                "properties": {
                    "node_type":  {"type": "string"},
                    "label":      {"type": "string"},
                    "edge_type":  {"type": "string"},
                    "from":       {"type": "string"},
                    "to":         {"type": "string"},
                },
            },
            "maxItems": 6,
        },
    },
    "required": ["indicator", "time_period", "unit", "population", "parent_path", "source_phrase"],
}


# ── List wrapper Schema — 한 문장 → N개 schema ─────────────────────
# [v6.13] 박재유 방식: 한 claim에서 여러 검증 가능 수치를 list로 반환.
CLAIM_SCHEMA_LIST_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schemas": {
            "type": "array",
            "description": (
                "이 문장에서 검증 가능한 *모든* 수치 주장 (각각 별도 schema). "
                "절대값과 비율이 같이 있으면 둘 다 추출. "
                "rank(순위) 표현 + 절대값 있으면 절대값만."
            ),
            "items": CLAIM_SCHEMA_JSON_SCHEMA,
            "minItems": 0,
            "maxItems": 5,
        },
    },
    "required": ["schemas"],
}
SCHEMA_INDUCTION_PROMPT = """당신은 뉴스 수치 주장에서 공식 통계 검증에 필요한 정보를 추출하는 데이터 엔지니어입니다.

═════════════════════════════════════════════════════════════════════════
[검증 대상 문장] ← 이 문장의 수치만 추출
"{claim_text}"
═════════════════════════════════════════════════════════════════════════

[문맥 — 참고용 ONLY] ← 이 안의 수치는 절대 추출 금지
{context}

도메인: {domain}
{domain_hint}
{temporal_hint}

[★ 가장 중요한 규칙: Context Leak 금지]
- 수치(value, source_phrase)는 *반드시* [검증 대상 문장]에 *literally* 등장하는 것만 추출
- [문맥]에 있는 수치는 *시점·주체 해소용 참고*만 (예: "지난해" → 2024, "이는" → 출생아 수)
- 검증 대상 문장에 수치가 *없으면* → 그 수치에 대한 schema는 *만들지 마세요*. 빈 리스트도 허용 (schemas=[])

[검증 대상 문장에 수치가 없는 예시]
문장: "이는 1991년 4월(8.1%) 이후 34년 만에 가장 높은 증가율이다."
  → 문장에 *literally* 있는 숫자: "1991", "4", "8.1", "34"
  → "1991", "4"는 시점/기간 표현 (value 아님)
  → "8.1"은 *과거 시점 비교 기준*이므로 검증 가능 (옵션: schema 1개)
  → "34"는 순위 표현이므로 value 아님
  → 결과: schemas=[{{"indicator": "출생아 수 증가율", "value": 8.1, "unit": "%", "time_period": "1991-04", "source_phrase": "8.1%", ...}}]
  → ★ 절대 *앞 문장*의 20171, 6.7을 가져오지 마세요. 이 문장에 없음.

[작업 목표]
[검증 대상 문장]에 검증 가능한 수치 주장이 여러 개 있을 수 있습니다.
각 수치 주장마다 별도 schema 객체로 추출하세요.
각 schema에 source_phrase (원문 문구)를 반드시 명시하세요.

[예시 — 한 문장에 2개 수치 (증가율 + 기준값)]
검증 대상 문장: "올해 4월 출생아 수는 2만 171명으로 지난해 같은 달(1만 9059명)보다 6.7% 늘었다"
결과:
  schemas: [
    {{indicator: "출생아 수", value: 20171, unit: "명", time_period: "2025-04",
      source_phrase: "2만 171명",
      parent_path: "인구 > 출생 > 출생아 수"}},
    {{indicator: "출생아 수 증가율", value: 6.7, unit: "%", time_period: "2025-04",
      source_phrase: "6.7%",
      prev_value: 19059, prev_time_period: "2024-04", prev_phrase: "1만 9059명",
      parent_path: "인구 > 출생 > 출생아 수"}}
  ]
  ★ 증가율 schema는 *기준값이 문장에 있으면* prev_value/prev_time_period/prev_phrase 의무.

[예시 — 한 문장에 1개 수치]
검증 대상 문장: "올해 합계출산율은 0.76명이다"
결과:
  schemas: [
    {{indicator: "합계출산율", value: 0.76, unit: "명", time_period: "2025",
      source_phrase: "0.76명",
      parent_path: "인구 > 출생 > 합계출산율"}}
  ]

[예시 — ★ 자연어 시점 표현의 month 추론]
검증 대상 문장: "2025년도 마지막 월의 출생아 수 증가율은 3.9%이다"
결과:
  schemas: [
    {{indicator: "출생아 수 증가율", value: 3.9, unit: "%", time_period: "2025-12",
      source_phrase: "3.9%",
      prev_value: null, prev_time_period: "2024-12", prev_phrase: "",
      parent_path: "인구 > 출생 > 출생아 수"}}
  ]
  ★ 핵심: "마지막 월" → 12월로 추론 → time_period="2025-12" (연 단위가 아님).
     prev_time_period="2024-12"도 "전년 동월" 규칙으로 연쇄 추론.
  ★ 만약 원문이 단순히 "2025년 출생아 수 증가율은 3.9%"였다면 (월 단서 없음)
     → time_period="2025" 으로 두고 month 추론 금지.

[예시 — rank 표현만 (검증 불가)]
검증 대상 문장: "출생아 수, 34년 만에 최대 증가"
결과:
  schemas: []   (절대 수치 없음, 순위 표현만 → 추출 X)

[예시 — 비교 기준값 + 변화량]
검증 대상 문장: "합계출산율 0.76명으로 지난해 같은 달보다 0.04명 증가"
결과:
  schemas: [
    {{indicator: "합계출산율", value: 0.76, unit: "명", time_period: "2025-04",
      source_phrase: "0.76명"}},
    {{indicator: "합계출산율 차이", value: 0.04, unit: "명", time_period: "2025-04",
      source_phrase: "0.04명",
      prev_value: 0.72, prev_time_period: "2024-04", prev_phrase: ""}}
  ]
  ★ "차이 0.04" schema의 prev_value=0.72 (계산: 0.76 - 0.04). prev_phrase는 *문장에 직접 없으면* 빈 문자열.

[예시 — ★ 다년 집계 (평균/총합/최대/최소 류)]
검증 대상 문장: "최근 3년간 평균 해외이주 신고 인원은 2904명"
결과:
  schemas: [
    {{indicator: "해외이주 신고 인원", value: 2904, unit: "명", time_period: "2024",
      source_phrase: "2904명",
      aggregation: "mean", aggregation_window: 3,
      aggregation_time_range: ["2022", "2023", "2024"],
      parent_path: "인구 > 이주 > 해외이주"}}
  ]
  ★★ 핵심: "평균/총합/최대/최소" 같은 집계 표현이 있으면 *aggregation 필드*를 채우세요.
     - aggregation: "mean" | "sum" | "max" | "min" | "median" (영어 소문자, 연산자만)
     - aggregation_window: "최근 N년/N분기/N개월" 의 정수 N (예: "최근 3년" → 3)
     - aggregation_time_range: 명시적으로 추론한 시점 리스트 (예: 2024 기준 "최근 3년" → ["2022","2023","2024"])
     - time_period: 가장 최근 시점 (단일 fetch 실패 시 fallback용)
     - prev_value/prev_time_period/prev_phrase는 *비워둠* (집계는 단일 prev 비교 아님)
  ★★ "최근 N년", "지난 N분기 평균", "총", "합계", "최대", "최저", "역대 가장 많은"
     등 다년/다기간 집계 신호가 있는 경우에만 aggregation 필드를 채우세요.
     집계가 아닌 단일 시점 값(예: "2024년 출생아 수")은 aggregation=null로 두세요.

[예시 — ★ 증가율인데 비교값이 본문에 없음 (시점만 계산)]
검증 대상 문장: "2023년 출생아 수는 23만 명으로 1년 전보다 7.7% 줄었다"
결과:
  schemas: [
    {{indicator: "출생아 수", value: 230000, unit: "명", time_period: "2023",
      source_phrase: "23만 명",
      parent_path: "인구 > 출생 > 출생아 수"}},
    {{indicator: "출생아 수 증가율", value: 7.7, unit: "%", time_period: "2023",
      source_phrase: "7.7%",
      prev_value: null, prev_time_period: "2022", prev_phrase: "",
      parent_path: "인구 > 출생 > 출생아 수"}}
  ]
  ★★ 핵심: "1년 전 출생아 수"가 본문에 숫자로 안 적혀 있음
     → prev_value=null (본문에 없으니까)
     → 하지만 prev_time_period="2022"는 *반드시 채움*
        ("2023년" + "1년 전" → 2022 로 계산).
     → 검증 단계에서 통계 DB의 2022년 출생아 수를 조회해
        (현재값 - 2022년값) 으로 7.7% 를 직접 계산·검증함.

[예시 — ★★ 동사형 변화량 표현 (도입/들여와/추가/늘어/줄어 + 증가율 결합)]
검증 대상 문장: "서울과 경기, 인천은 각각 652개, 592개, 169개의 장비를 들여와 5.5%~6.5%의 증가세를 보인 반면, 강원도는 단 52개로 4% 증가에 그쳤다"
결과:
  schemas: [
    {{indicator: "의료장비 증가 수", value: 652, unit: "대", time_period: "2024",
      population: "서울", source_phrase: "652개",
      parent_path: "보건 > 의료자원 > 의료장비"}},
    {{indicator: "의료장비 증가 수", value: 592, unit: "대", time_period: "2024",
      population: "경기", source_phrase: "592개"}},
    {{indicator: "의료장비 증가 수", value: 169, unit: "대", time_period: "2024",
      population: "인천", source_phrase: "169개"}},
    {{indicator: "의료장비 증가 수", value: 52, unit: "대", time_period: "2024",
      population: "강원도", source_phrase: "52개"}},
    {{indicator: "의료장비 증가율", value: 6.0, unit: "%", time_period: "2024",
      population: "서울", source_phrase: "5.5%~6.5%",
      modifier: "범위(5.5~6.5)의 중앙값"}},
    {{indicator: "의료장비 증가율", value: 4.0, unit: "%", time_period: "2024",
      population: "강원도", source_phrase: "4%"}}
  ]
  ★★ 핵심: 문장에 *동사*("들여와", "도입했다", "추가했다", "신규로 늘렸다", "줄어들었다")가
     있고 그 옆에 *개수*와 *%*가 함께 나오면 → *두 가지 schema*로 분기:
     (1) "<원지표> 증가 수" (또는 감소 수) — value는 변화 *개수*
     (2) "<원지표> 증가율" (또는 감소율) — value는 변화 *%*
  ★★ value를 단순히 *총 보유 수*로 해석하면 안 됨.
     "652개의 장비를 들여와" → 652는 *신규 도입 수*이지 *총 보유 수*가 아님.
  ★★ 범위 표현("5.5%~6.5%")이 있으면 *중앙값*을 value로, 원문은 source_phrase + modifier로.

[예시 — ★★ 동사로만 표현된 변화 (수치는 한쪽만)]
검증 대상 문장: "지난해 입국자가 30만 명 늘어나 전년 대비 8% 증가했다"
결과:
  schemas: [
    {{indicator: "입국자 증가 수", value: 300000, unit: "명", time_period: "2024",
      source_phrase: "30만 명",
      parent_path: "인구 > 이주 > 입국자"}},
    {{indicator: "입국자 증가율", value: 8.0, unit: "%", time_period: "2024",
      source_phrase: "8%",
      prev_value: null, prev_time_period: "2023", prev_phrase: ""}}
  ]
  ★★ "X 늘어나 Y%" 패턴 → *두 schema 모두* 생성 (증가 수 + 증가율).
     X(30만 명)는 *총량 아닌 변화량* — indicator에 "증가 수" 명시 필수.

[예시 — ★ 한 문장에 지역별 수치 나열 (같은 지표, 지역만 다름)]
검증 대상 문장: "동작구가 10.6%로 가장 높았다. 이어 성동구(8.9%), 마포구(8.7%), 영등포구(7.9%)"
결과:
  schemas: [
    {{indicator: "표준주택 공시가격 변동률", value: 10.6, unit: "%", time_period: "2020",
      population: "동작구", source_phrase: "10.6%",
      parent_path: "주택 > 공시가격 > 변동률"}},
    {{indicator: "표준주택 공시가격 변동률", value: 8.9, unit: "%", time_period: "2020",
      population: "성동구", source_phrase: "8.9%",
      parent_path: "주택 > 공시가격 > 변동률"}},
    {{indicator: "표준주택 공시가격 변동률", value: 8.7, unit: "%", time_period: "2020",
      population: "마포구", source_phrase: "8.7%",
      parent_path: "주택 > 공시가격 > 변동률"}},
    {{indicator: "표준주택 공시가격 변동률", value: 7.9, unit: "%", time_period: "2020",
      population: "영등포구", source_phrase: "7.9%",
      parent_path: "주택 > 공시가격 > 변동률"}}
  ]
  ★★ 핵심: "동작구(10.6%)", "성동구(8.9%)" 처럼 *지역+수치 쌍이 여러 개*면
     각 쌍을 *반드시 개별 schema*로. 절대 "자치구별 상승률" 하나로 뭉뚱그리지 마세요.
  ★★ 지역명은 population에, 수치는 value에 각각 채웁니다. value를 null로 두지 마세요.
  ★★ indicator는 모든 schema가 동일 (지표는 같고 지역만 다르므로).

[핵심 규칙]

1. **단위 통일**: 한글 단위는 정확하게 숫자로 변환. *'만'은 10,000*.
   · "2만 171명"  → value=20171 (NOT 2,171,000)
   · "1만 9059명" → value=19059
   · "21만 7천명" → value=217000
   · "1만 7921건" → value=17921
   · "23만 8000명" → value=238000, source_phrase="23만 8000명" (NOT "23만 8천명")
   · "19만 3000건" → value=193000, source_phrase="19만 3000건" (NOT "19만 3천건")
   · ★ source_phrase는 원문 그대로. 절대 한자어로 바꾸지 마세요 (8000 → 8천 금지)
   · "2.2%였다" → value=2.2, unit="%", source_phrase="2.2%"
   · "~였다/~이다/~다" 뒤에 오는 수치도 추출 대상

2. **★ 한 문장에 여러 수치**: 각각 별도 schema. *놓치지 마세요*.
   · "X명으로 Y% 늘었다" → 2개 schema (절대값 + 비율)
   · "X로 Z 증가" → 2개 schema (현재값 + 변화량)
   · **★ 지역(항목)별 나열** "A구(10.6%), B구(8.9%), C구(8.7%)"
     → 각 지역마다 별도 schema. indicator는 동일, population=지역명,
        value=각 수치. 절대 하나로 합치거나 value=null로 두지 마세요.
   · 이 규칙은 지역뿐 아니라 연령대·업종·품목 등 *모든 분류 축*에 적용.

2-1. **★★ 동사형 변화 표현 — *총량 vs 변화량* 구분이 핵심**:
   문장에 *변화/증감을 나타내는 동사*(들여와, 도입했다, 추가됐다, 신규로,
   늘어, 줄어, 감소, 증가, 매입, 신축, 폐기 등)가 있으면, 그 옆 수치는
   *총량이 아니라 변화량*임. indicator에 "증가 수"/"감소 수"/"신규 도입 수"
   등 변화 의미를 *명시*해야 함:
   · "X개 들여와" → indicator="<원지표> 증가 수" (or "신규 도입 수"), value=X
   · "X명 늘어" → indicator="<원지표> 증가 수", value=X
   · "X건 줄어" → indicator="<원지표> 감소 수", value=X
   같은 문장에 % 표현이 함께 있으면 *증가율 schema*도 별도 생성:
   · "X개 들여와 Y% 증가세" → 2개 schema (증가 수=X + 증가율=Y)
   ★ 절대 X를 *총량*으로 해석하지 마세요. "652개 들여와" → 652는 *신규 도입*.
   ★ 범위 표현(`5.5%~6.5%`)이 있으면 *중앙값*을 value로, 원문은 source_phrase + modifier로.

3. **★ value/unit/source_phrase 일관성**:
   · source_phrase가 "X%" → unit="%", value는 비율
   · source_phrase가 "X명/건/원" → unit=명/건/원, value는 절대값

4. **단위 강제**: unit 절대 비우지 마세요. 불분명하면: 지수/배/위/점.

5. **indicator는 측정 대상 자체**:
   ✗ "출생아 수 증가율" — 단, *별도 schema*로 % 추출 시엔 허용
   ○ 절대값 schema: indicator="출생아 수"
   ○ 비율 schema:   indicator="출생아 수 증가율" 또는 indicator="출생아 수" + unit="%"

6. **분류 축은 population**:
   ✗ indicator="대졸이상 쉬었음 청년"
   ○ indicator="쉬었음 인구", population="대졸 이상 청년"

7. **parent_path**: "대분류 > 중분류 > 소분류". 기사 제목/출처/기관명 금지.
   KOSIS 대분류: 인구/가구/고용/노동/임금/물가/가계/보건/사회/복지/교육/
   환경/농림/수산/건설/주택/토지/교통/정보통신/경제/산업/무역.

8. **시점 형식**: "YYYY" 또는 "YYYY-MM"만. "2024년 4월" → "2024-04".

9. **순위 표현 단독**: "N년 만에 최대" 단독이면 그 N은 value 아님.

10. **근사 표현**: "안팎/이상/약/가량" 있으면 is_approximate=true, modifier=원문.

11. **★ source_phrase 의무**: 모든 schema는 source_phrase 필드를 *반드시* 포함.
    이 문구는 [검증 대상 문장]에 *literally* 등장해야 함. 등장하지 않으면 schema 추출 금지.
"""

# ── [2026-05-27] regenerate_schema ───────────────────────────────────
# replan tool용 — 초기 induction이 잘못 분류한 schema를 *표 row sample 보고* 재분류.
#
# 배경: schema_inductor의 초기 호출은 *원문 텍스트만* 보고 schema를 만듦. 그래서:
#   - "강원도 의료장비 *증가 수* 52" 같은 동사형 변화량을 *base*(absolute)로 잘못 분류
#   - 표 row에 "52"라는 절대값 row가 없음에도 LLM이 absolute로 처리
# replan 시점엔 *실제 표 데이터*까지 봤기 때문에, "52는 row로 안 들어있고 표엔
# 절대값 N대만 있음"을 LLM에 보여주면 → "이건 delta"로 재분류 가능.
#
# 호출 위치: structverify.agent.tools.replan.ReplanTool
# 결과: 새 schema dict (또는 None). 호출자는 이걸로 claim.schema 업데이트 후
#       planner.plan() 재호출.

REGENERATE_SCHEMA_PROMPT = """당신은 통계 검증 schema 수정자입니다.

초기 schema induction이 원문만 보고 만든 schema가 실제 표 데이터와 *구조적으로
맞지 않는다*는 게 드러났습니다. 표의 row sample을 보고 **schema를 재분류**하세요.

[원문 claim]
{claim_text}

[현재 schema — 초기 induction 결과]
{original_schema_json}

[실제 표에서 받은 데이터 — observation summary]
{observations_summary}

[★ 재분류 룰 — 매우 중요]
1. claim.value가 표의 *row에 직접 매칭되는 값*인가?
   - YES → value_role="base" (absolute claim). 그대로 유지.
   - NO + 표에 *해당 indicator의 원시 절대값* 존재 → 계산 필요:
     · indicator 접미사가 "증가율/감소율/증감률/변동률" → value_role="derived_rate"
     · indicator 접미사가 "증가 수/감소 수/증감/변화량/늘었/줄었" → value_role="derived_difference"
   - NO + 표에 *집계 단서*(다년 평균/합계 등) → value_role="aggregation"

2. value_role을 *base가 아닌 것*으로 바꾸면, prev_time_period 채우기:
   - time_period 단위가 연(예: '2024')이면 → prev_time_period = "전년" (예: '2023')
   - 월(예: '2024-04')이면 → prev_time_period = "전년 동월" (예: '2023-04')
   - 분기는 전년 동분기.

3. 표에 *claim 시점 데이터가 없음*이 확실하면 (예: 표 PRD_DE 분포가 2021-2023뿐인데
   claim time_period='2024') → 그건 schema 문제가 아닌 *데이터 부재* 문제.
   schema 재분류로 해결 안 됨. 이 경우엔 value_role 그대로 두고 notes에 명시.

[★ 금지]
- value, unit, indicator는 원문 의도와 같으면 *변경 금지*.
- 표 row에 직접 매칭되는 row가 있는데 derived로 바꾸는 것 금지 (no-op 회피).
- 단순 retry plan을 만들기 위한 schema "복제"는 금지 — 실제로 *구조*가 바뀌어야 한다.

[출력 형식 — JSON only, 다른 텍스트 금지]
{{
  "indicator": "...",
  "time_period": "...",
  "unit": "...",
  "population": "...",
  "value": <number>,
  "value_role": "base | derived_rate | derived_difference | aggregation",
  "prev_value": <number or null>,
  "prev_time_period": "<YYYY 또는 YYYY-MM 또는 null>",
  "prev_phrase": "<원문에서 직접 인용한 prev 단서 또는 ''>",
  "parent_path": "...",
  "modifier": "<범위/근사 표현 등 or null>",
  "reason": "<왜 이렇게 재분류했는지 한 줄>"
}}
"""
