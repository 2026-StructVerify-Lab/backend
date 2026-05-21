"""
detection/schema_inductor.py — Dynamic Schema Induction (Step 5)

[김예슬 - 2026-04-22]
- SCHEMA_INDUCTION_PROMPT: 도메인 컨텍스트 주입 + 예시 강화
- _safe_float(): 다양한 수치 표현 파싱 ("64.2%", "약 64" 등)
- _validate_schema(): 최소 유효성 검증
- 재시도 로직 추가 (최대 2회)

[김예슬 - 2026-04-24]
- generate_json() → generate_structured() 으로 교체
  · Structured Outputs (HCX-007) → JSON Schema 보장 (파싱 실패 없음)
- CLAIM_SCHEMA_JSON_SCHEMA: ClaimSchema에 대응하는 JSON Schema 정의 추가

[v6.11 - 2026-05-12]
- 룰베이스 후처리 제거 (_cleanse_indicator, _normalize_time_period)
- 박재유 SYSTEM_PROMPT 스타일 차용: 단위 강제 + indicator 정제 + parent_path 추출
- ClaimSchema 신규 필드 추출: parent_path / is_approximate / modifier
- 모든 정제 책임은 LLM에게 위임 (룰 베이스 X)

# [박재윤 - 2026-05-14]: SCHEMA_INDUCTION_PROMPT system_prompt 개선
#   · 예보/예상/전망/예측 indicator → schema 추출 금지 규칙 추가

# [박재윤 - 2026-05-15]: SCHEMA_INDUCTION_PROMPT 수치 추출 규칙 보강
#   · "~였다/~이다/~다" 패턴 수치도 추출 대상 명시 (근원물가 2.2% 누락 방지)
#   · "N만 M천" 복합 단위 패턴 _extract_numbers_from_text에 추가
#     (24만 2천 → 242000 환산 오류 방지)

# [박재윤 - 2026-05-18]: SCHEMA_INDUCTION_PROMPT source_phrase 원문 보존 규칙 추가
#   · "23만 8000명" → source_phrase 원문 그대로 (8000→8천 변환 금지)

# [박재윤 - 2026-05-18]: _extract_numbers_from_text "N만 NNNN" 패턴 추가
#   · "2869만 3000명" → 28693000 환산 (4자리 숫자 붙는 패턴)
"""
from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from structverify.core.schemas import Claim, ClaimSchema
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


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
                "예: '2024', '2024-10'. 한글 표기('2024년 10월') 금지."
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


# ── 메인 진입점 ────────────────────────────────────────────────────────
async def induce_schemas(
    claims: list[Claim],
    config: dict | None = None,
    graph: "ClaimGraph | None" = None,
) -> list[Claim]:
    """
    각 주장에서 ClaimSchema들을 동적으로 유도한다.

    [v6.13] 한 claim에서 *여러 ClaimSchema* 추출 가능 (박재유 방식).
    LLM이 한 문장의 모든 검증 가능 수치를 list로 반환.
    첫 schema는 원래 claim에 부착, 나머지는 claim 복제해서 부착.

    [v6 멀티홉] graph 있으면 시점 해소 결과를 prompt hint로 주입.
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))
    success, fail = 0, 0
    expanded: list[Claim] = []

    for claim in claims:
        domain = config.get("detected_domain", "general")
        domain_hint = (
            f"주요 지표 예시: {DOMAIN_HINTS[domain]}"
            if domain in DOMAIN_HINTS else ""
        )

        context = getattr(claim, "context_text", None) or claim.claim_text
        temporal_hint = _build_temporal_hint(graph, claim) if graph else ""
        # [v6.16] 시점 표현이 전혀 없는 claim의 fallback 기준 연도
        anchor_year = graph.get_anchor_year() if graph else None

        schemas = await _induce_multiple(
            llm, claim.claim_text, domain, domain_hint,
            context=context, temporal_hint=temporal_hint,
            anchor_year=anchor_year,
        )

        if not schemas:
            # 검증 가능 수치 0개 — 원래 claim은 유지하되 schema=None
            fail += 1
            expanded.append(claim)
            logger.warning(
                f"스키마 유도: {claim.sent_id} → 검증 가능 수치 없음"
            )
            continue

        # [v6.17] value=null 중복 schema 제거
        #   LLM이 value를 못 채우고 indicator만 같은 빈 schema를 N개 만드는
        #   경우만 정리. 단, population까지 같아야 진짜 중복으로 간주.
        #   ★ "동작구 10.6%, 성동구 8.9%"처럼 지역만 다른 정상 다중 수치는
        #     population이 다르므로 합쳐지지 않음 (이전엔 다 뭉개지던 버그).
        deduped: list[ClaimSchema] = []
        seen_null_keys: set[tuple] = set()
        for sch in schemas:
            if sch.value is None:
                key = (
                    sch.indicator or "",
                    sch.time_period or "",
                    sch.population or "",   # ★ population 추가 — 지역별 구분
                )
                if key in seen_null_keys:
                    logger.info(
                        f"  [중복 제거] value=null 중복 schema 폐기 "
                        f"(indicator={sch.indicator}, population={sch.population})"
                    )
                    continue
                seen_null_keys.add(key)
            deduped.append(sch)
        schemas = deduped

        # 첫 schema는 원래 claim에 부착
        claim.schema = schemas[0]
        expanded.append(claim)
        success += 1
        logger.info(
            f"스키마 유도: {claim.sent_id} [1/{len(schemas)}] "
            f"indicator={schemas[0].indicator}, value={schemas[0].value}, "
            f"unit={schemas[0].unit}, time_period={schemas[0].time_period}, "
            f"parent_path={schemas[0].parent_path}"
        )

        # 추가 schema들은 claim 복제 후 부착 (claim_id 새로 발급)
        for i, sch in enumerate(schemas[1:], start=2):
            cloned = claim.model_copy(update={
                "claim_id": uuid4(),
                "schema": sch,
            })
            expanded.append(cloned)
            success += 1
            logger.info(
                f"스키마 유도: {claim.sent_id} [{i}/{len(schemas)}] (복제) "
                f"indicator={sch.indicator}, value={sch.value}, "
                f"unit={sch.unit}, time_period={sch.time_period}, "
                f"parent_path={sch.parent_path}"
            )

    logger.info(
        f"스키마 유도 완료: {len(claims)}개 claim → {len(expanded)}개 claim "
        f"(성공 schema {success}건, 실패 claim {fail}건)"
    )
    return expanded


# [v6.20] claim 문장 텍스트에서 직접 셀 상대/절대 시점 표현 패턴.
# document_graph의 LLM temporal agent가 한 문장의 표현을 일부 누락하면
# (예: "작년...재작년..."에서 "작년"을 빠뜨림) count_temporal_expressions가
# 1을 반환 → multi_temporal=False → 잘못된 단정 hint. 그래서 그래프와
# 별개로 claim_text를 정규식으로 스캔해 보수적으로 multi 여부를 판정한다.
_TEMPORAL_TEXT_PATTERNS = (
    "재작년", "지지난해", "지지난 해",
    "작년", "지난해", "지난 해", "전년",
    "올해", "금년", "이번 해",
    "내년", "이듬해", "다음 해",
    "내후년",
)


def _count_temporal_in_text(text: str) -> int:
    """claim 문장 텍스트에서 상대 시점 표현의 개수를 센다.

    "작년 X도, 재작년 Y도" → 2 (작년 1 + 재작년 1).
    겹치는 패턴 중복 카운트를 막기 위해, 긴 패턴부터 매칭하며
    매칭된 구간을 소거한다 ('재작년'을 먼저 잡아야 '작년'이
    그 안에서 다시 안 잡힌다).
    """
    if not text:
        return 0
    s = str(text)
    count = 0
    # 긴 패턴 우선 (재작년 → 작년 순서 보장)
    for pat in sorted(_TEMPORAL_TEXT_PATTERNS, key=len, reverse=True):
        while pat in s:
            count += 1
            s = s.replace(pat, "\x00" * len(pat), 1)  # 매칭 구간 소거
    return count


def _build_temporal_hint(graph: "ClaimGraph", claim: Claim) -> str:
    """
    그래프 시점 해소 결과를 prompt hint 텍스트로.

    [v6.15] 상대 시점 표현 매핑 강화:
      anchor_year 기준으로 '내년/올해/작년/지난해/재작년'을 모두 절대 연도로
      변환하는 표를 LLM에게 명시 → time_period=null 방지.
    """
    prov = graph.temporal_provenance(claim)
    anchor_year = graph.get_anchor_year()

    # [v6.19] 한 문장에 시간표현이 여러 개면 (예: "작년 X도, 재작년 Y도")
    # temporal_provenance가 어느 표현이 이 claim의 것인지 구분 못 하고
    # 첫 번째를 무조건 반환한다 → 단정적 hint가 틀릴 수 있음.
    # 이 경우 단정하지 말고 anchor 변환표만 줘서 LLM이 claim 문맥으로
    # 직접 시점을 고르게 한다.
    _te_count = graph.count_temporal_expressions(claim)
    # [v6.20] 그래프 카운트와 별개로 claim 문장 텍스트도 직접 스캔.
    # temporal agent가 표현을 누락해도(그래프 te_count=1) 텍스트에
    # 상대표현이 2개 이상이면 multi로 판정 → 잘못된 단정 hint 방지.
    _text_te_count = _count_temporal_in_text(
        getattr(claim, "claim_text", "") or ""
    )
    multi_temporal = (_te_count > 1) or (_text_te_count > 1)

    # [v6.19 진단] multi_temporal 판정과 분기 결정을 로그로 — 어느 경로를
    # 탔는지 안 보여서 temporal 수정이 먹혔는지 확인이 안 됨.
    _branch = (
        "단정(prov)" if (prov and prov.get("resolved") and not multi_temporal)
        else ("변환표(anchor)" if anchor_year is not None else "없음")
    )
    logger.info(
        f"[temporal_hint] {getattr(claim, 'sent_id', '?')}: "
        f"te_count={_te_count} text_te={_text_te_count} "
        f"multi_temporal={multi_temporal} "
        f"prov_resolved={(prov or {}).get('resolved')} "
        f"prov_expr={(prov or {}).get('expression')!r} "
        f"anchor={anchor_year} → branch={_branch}"
    )

    if prov and prov.get("resolved") and not multi_temporal:
        return (
            f"\n[시점 정보 — 그래프 해소 결과]\n"
            f"- 원문 표현: {prov.get('expression')}\n"
            f"- 해소된 절대 시점: {prov['resolved']}\n"
            f"- 근거: {prov.get('basis') or '문서 anchor 기반'}\n"
            f"위 절대 시점을 time_period로 사용하세요."
        )
    elif anchor_year is not None:
        # [v6.15] 상대 표현 → 절대 연도 변환표를 명시적으로 제공
        multi_note = ""
        if multi_temporal:
            # [v6.19] 한 문장에 시간표현이 여러 개 — 수치별로 구분 지시
            multi_note = (
                f"- ⚠️ 이 문장에는 시점 표현이 *둘 이상* 있습니다 "
                f"(예: '작년 X도, 재작년 Y도').\n"
                f"  각 수치 바로 앞/근처의 시점 표현을 보고 schema마다 "
                f"time_period를 *개별적으로* 정확히 매칭하세요.\n"
                f"  모든 수치에 같은 시점을 쓰지 마세요.\n"
            )
        return (
            f"\n[시점 정보 — 문서 anchor]\n"
            f"- 이 문서의 기준 연도(anchor_year): {anchor_year}\n"
            f"{multi_note}"
            f"- 상대 시점 표현은 *반드시* 아래 표대로 절대 연도로 변환하세요:\n"
            f"    '내후년'        → {anchor_year + 2}\n"
            f"    '내년/이듬해'   → {anchor_year + 1}\n"
            f"    '올해/금년/현재' → {anchor_year}\n"
            f"    '작년/지난해'   → {anchor_year - 1}\n"
            f"    '재작년'        → {anchor_year - 2}\n"
            f"- ★ 검증 대상 문장에 위 상대 표현이 하나라도 있으면\n"
            f"  time_period를 절대 연도(예: '{anchor_year + 1}')로 *반드시* 채우세요.\n"
            f"- ★ time_period를 null로 두지 마세요. 시점 단서가 전혀 없을 때만 null."
        )
    return ""


async def _induce_multiple(
    llm: LLMClient,
    claim_text: str,
    domain: str = "general",
    domain_hint: str = "",
    context: str = "",
    temporal_hint: str = "",
    anchor_year: int | None = None,
) -> list[ClaimSchema]:
    """
    단일 주장 → list[ClaimSchema] (0개 이상).

    LLM이 한 문장의 모든 검증 가능 수치를 schemas 배열로 반환.
    [v6.14] source_phrase 검증으로 context leak 방지:
      - LLM이 schema마다 source_phrase 제공 (예: '2만 171명', '6.7%')
      - source_phrase가 claim_text에 *substring으로 등장*하는지 검증
      - 등장하지 않으면 (context leak) 그 schema 폐기
    Structured Outputs 사용 — JSON 파싱 실패 없음.
    """
    prompt = SCHEMA_INDUCTION_PROMPT.format(
        claim_text=claim_text,
        context=context or claim_text,
        domain=domain,
        domain_hint=domain_hint,
        temporal_hint=temporal_hint,
    )

    try:
        r = await llm.generate_structured(
            prompt=prompt,
            schema=CLAIM_SCHEMA_LIST_JSON_SCHEMA,
            system_prompt=(
                "통계 분석 전문가. 위 규칙을 엄격히 따르세요. "
                "★ 핵심: [검증 대상 문장]에 literally 등장하는 수치만 추출. "
                "[문맥]의 수치는 절대 추출 금지. "
                "각 schema에 source_phrase 의무 포함. "
                "★ '예상/예보/전망/예측' 포함 indicator는 KOSIS 검증 불가 → 해당 schema 추출 금지."
            ),
        )
    except Exception as e:
        logger.warning(f"스키마 유도 LLM 호출 예외: {e}")
        return []

    schemas_raw = r.get("schemas") or []
    if not isinstance(schemas_raw, list):
        logger.warning(f"스키마 유도: schemas가 list 아님 ({type(schemas_raw)})")
        return []

    results: list[ClaimSchema] = []
    for item in schemas_raw:
        if not isinstance(item, dict):
            continue

        # ── [v6.14] source_phrase 검증 (context leak 방지) ──
        source_phrase = (item.get("source_phrase") or "").strip()
        if source_phrase:
            if not _source_phrase_in_claim(source_phrase, claim_text):
                logger.warning(
                    f"  ⚠️ context leak 감지: source_phrase={source_phrase!r} "
                    f"가 검증 대상 문장에 없음 → schema 폐기 "
                    f"(indicator={item.get('indicator')}, value={item.get('value')})"
                )
                continue
        else:
            # source_phrase 없으면 value 자체로 검증 (LLM이 의무 위반한 경우 fallback)
            val = item.get("value")
            if val is not None and not _value_in_claim_text(val, claim_text):
                logger.warning(
                    f"  ⚠️ context leak 의심 (source_phrase 누락): "
                    f"value={val} 가 문장에 없음 → schema 폐기 "
                    f"(indicator={item.get('indicator')})"
                )
                continue

        try:
            # [v6.14 E fix] value 환산 정확성 검증 + 자동 교정
            # LLM이 "2만 171" → 21710 같은 환산 오류를 내는 경우 발견됨.
            # source_phrase가 있으면 거기서 직접 환산값 추출 → LLM value와 비교 → 5% 이상 차이면 교정.
            raw_value = _safe_float(item.get("value"))
            corrected_value, was_corrected = _verify_and_correct_value(
                raw_value, source_phrase
            )
            if was_corrected:
                logger.warning(
                    f"  🔧 value 환산 교정: LLM={raw_value} → 교정={corrected_value} "
                    f"(source_phrase={source_phrase!r}, indicator={item.get('indicator')})"
                )

            # [v6.14 C2] prev_value 검증 + ClaimSchema 생성
            # schemas.py에 prev_value 필드가 *아직 없는 경우* import 안 깨지게 try/except
            prev_value_raw = item.get("prev_value")
            prev_value = _safe_float(prev_value_raw) if prev_value_raw is not None else None
            prev_time_period = (item.get("prev_time_period") or "").strip() or None
            prev_phrase = (item.get("prev_phrase") or "").strip()

            # prev_phrase가 있으면 검증 (context leak 방지 — source_phrase와 동일 규칙)
            # [v6.17] prev_value/prev_phrase만 폐기하고 prev_time_period는 유지.
            #   시점은 본문 표현("1년 전")에서 계산한 것이라 leak이 아니며,
            #   prev_value가 없어도 시점만 있으면 통계 DB에서 그 시점 값을
            #   직접 조회해 검증할 수 있음.
            if prev_phrase and not _source_phrase_in_claim(prev_phrase, claim_text):
                logger.warning(
                    f"  ⚠️ prev_phrase context leak: {prev_phrase!r} 가 검증 대상 문장에 없음 "
                    f"→ prev_value만 폐기 (prev_time_period={prev_time_period!r}는 유지, "
                    f"indicator={item.get('indicator')})"
                )
                prev_value = None
                prev_phrase = None
                # prev_time_period는 일부러 유지 — 검증 단계에서 사용

            # prev_phrase가 있으면 prev_value 환산 정확성도 검증 (E fix 응용)
            if prev_phrase and prev_value is not None:
                corrected_prev, prev_was_corrected = _verify_and_correct_value(
                    prev_value, prev_phrase
                )
                if prev_was_corrected:
                    logger.warning(
                        f"  🔧 prev_value 환산 교정: LLM={prev_value} → 교정={corrected_prev} "
                        f"(prev_phrase={prev_phrase!r})"
                    )
                prev_value = corrected_prev

            # [v6.16] time_period가 비어있으면 문서 anchor_year로 채움
            #   "전국 공시가격 상승률 4.5%" 처럼 시점 표현이 없는 문장도
            #   기사 작성연도(anchor_year) 기준으로 검증되도록 보정.
            _tp = item.get("time_period") or None
            if not _tp and anchor_year is not None:
                _tp = str(anchor_year)
                logger.info(
                    f"  [시점 보정] time_period 없음 → anchor_year={anchor_year} 적용 "
                    f"(indicator={item.get('indicator')})"
                )

            schema_kwargs = dict(
                indicator=item.get("indicator") or None,
                time_period=_tp,
                unit=item.get("unit") or None,
                population=item.get("population") or None,
                value=corrected_value,
                source_reference=item.get("source_reference") or None,
                graph_schema_candidates=item.get("graph_schema_candidates") or [],
                parent_path=item.get("parent_path") or None,
                is_approximate=bool(item.get("is_approximate", False)),
                modifier=item.get("modifier") or None,
            )
            # prev_* 필드는 schemas.py에 *추가됐을 때만* 전달
            # (구버전 schemas.py와 backward compat)
            try:
                ClaimSchema.model_fields["prev_value"]  # 필드 존재 여부 확인
                schema_kwargs["prev_value"] = prev_value
                schema_kwargs["prev_time_period"] = prev_time_period
                schema_kwargs["prev_phrase"] = prev_phrase or None
            except KeyError:
                # schemas.py에 prev_* 필드 없음 — 무시
                if prev_value is not None:
                    logger.warning(
                        f"  ℹ️ prev_value={prev_value} 추출됐으나 ClaimSchema에 필드 없음. "
                        f"core/schemas.py에 prev_value/prev_time_period/prev_phrase 필드 추가 필요."
                    )

            # [2026-05-21] value_role 자동 추론 — schema_inductor가 분기한 *이유*를
            # downstream planner에 명시적으로 전달. LLM이 같은 claim_text를 보고
            # base/derived를 헷갈리는 걸 방지.
            #
            # [K 패치 2026-05-21] indicator suffix 우선 검사. prev_value 유무는
            #   2차 신호로 격하. 합계출산율 0.79처럼 LLM이 *base 절대값 schema*에
            #   prev_value=0.73을 추가 정보로 박아도 *indicator에 ~증가/~차이 같은
            #   derived suffix가 없으면 base*로 분류.
            #
            #   - indicator suffix(~증가율/~비율 류) + 비율 단위 → derived_rate
            #   - indicator suffix(~증가/~감소/~차이 류, 비율 아님)    → derived_difference
            #   - 그 외 → base (prev_value 있어도 base — 단일 값 검증)
            try:
                ClaimSchema.model_fields["value_role"]
                _ind = (item.get("indicator") or "").strip()
                _unit = (item.get("unit") or "").strip()
                _RATE_SUFFIXES = (
                    "증가율", "감소율", "증감률", "변화율", "상승률", "하락률",
                    "비율", "비중",
                )
                _DIFF_SUFFIXES = (
                    "증가", "감소", "증감", "변화", "차이",
                )
                _is_rate_indicator = any(_ind.endswith(s) for s in _RATE_SUFFIXES)
                _is_pct_unit = _unit in ("%", "퍼센트", "퍼센트포인트", "%p")
                _is_diff_indicator = (
                    any(_ind.endswith(s) for s in _DIFF_SUFFIXES)
                    and not _is_rate_indicator
                )
                if _is_rate_indicator or _is_pct_unit:
                    schema_kwargs["value_role"] = "derived_rate"
                elif _is_diff_indicator:
                    schema_kwargs["value_role"] = "derived_difference"
                else:
                    # base — indicator suffix가 derived가 아니면 prev_value 유무
                    # 무관하게 base. prev_value는 후처리에서 *clear* 해서 reflect
                    # LLM이 자율 prev fetch 시도하지 않도록 한다.
                    schema_kwargs["value_role"] = "base"
                    if schema_kwargs.get("prev_value") is not None or schema_kwargs.get("prev_time_period"):
                        logger.info(
                            f"  [K] base 분류 → prev_value/prev_time_period clear "
                            f"(indicator={_ind!r}, was prev_value={schema_kwargs.get('prev_value')!r})"
                        )
                        schema_kwargs["prev_value"] = None
                        schema_kwargs["prev_time_period"] = None
                        schema_kwargs["prev_phrase"] = None
            except KeyError:
                pass  # value_role 필드 없는 구버전 — 무시

            schema = ClaimSchema(**schema_kwargs)
        except Exception as e:
            logger.debug(f"개별 schema 파싱 실패: {e}")
            continue

        if _validate_schema(schema):
            results.append(schema)

    # ── [v6.15 L fix] 차이 schema의 prev_value 자동 역산 ─────────────────
    # 같은 sentence에서 *절대값 schema*와 *차이 schema*가 함께 나왔을 때,
    # 차이 schema의 prev_value가 *비어있으면* → 절대값 − 차이값으로 역산.
    #
    # 예: "합계출산율 0.79명으로 지난해보다 0.06명 증가"
    #   - 절대값 schema: 합계출산율=0.79
    #   - 차이 schema: 합계출산율 차이=0.06, prev_value=None
    #   → 자동 역산: prev_value = 0.79 - 0.06 = 0.73
    #
    # 효과: verifier C2 분기가 작동 → KOSIS 절대값 row와 자동 계산 비교.

    def _prev_year_period(tp: str | None) -> str | None:
        """[수정 v6.23] time_period에서 '1년 전' 시점을 계산.

        '차이/증감' claim의 비교 기준은 보통 '지난해 같은 달/기간'이다.
        prev_time_period를 현재 시점과 똑같이 두면(이전 버그) prev와
        current가 같아져 → fetch가 두 시점을 못 가져오고 검증 불가.
        연도만 1 빼고 월/분기 부분은 그대로 둔다. 도메인 무관.

        '2025-04' → '2024-04'  /  '2023' → '2022'  /  '2025Q2' → '2024Q2'
        """
        if not tp:
            return None
        s = str(tp).strip()
        import re as _re
        m = _re.match(r"^(\d{4})(.*)$", s)
        if not m:
            return None
        try:
            year = int(m.group(1))
        except ValueError:
            return None
        return f"{year - 1}{m.group(2)}"

    try:
        _has_prev_field = "prev_value" in ClaimSchema.model_fields
    except Exception:
        _has_prev_field = False

    if _has_prev_field and len(results) >= 2:
        # 절대값 schema (indicator에 "차이/증감/변화량" 없음) 찾기
        abs_schemas = [
            s for s in results
            if s.indicator and not any(
                kw in s.indicator for kw in ("차이", "증감", "변화량", "증가율")
            ) and s.value is not None
        ]
        diff_schemas = [
            s for s in results
            if s.indicator and ("차이" in s.indicator or "증감" in s.indicator
                                or "변화량" in s.indicator)
            and s.value is not None
            and getattr(s, "prev_value", None) is None
        ]

        for diff_s in diff_schemas:
            # 같은 indicator base 찾기 (예: "합계출산율 차이" → "합계출산율")
            diff_base = diff_s.indicator
            for kw in ("차이", "증감", "변화량"):
                diff_base = diff_base.replace(kw, "").strip()

            # 매칭되는 절대값 schema
            matching_abs = None
            for abs_s in abs_schemas:
                if abs_s.indicator and (abs_s.indicator == diff_base
                                        or diff_base in abs_s.indicator
                                        or abs_s.indicator in diff_base):
                    # 단위도 비슷한지 (둘 다 비어있거나 둘 다 있고 같은 type)
                    if (not abs_s.unit and not diff_s.unit) or \
                       (abs_s.unit and diff_s.unit and abs_s.unit == diff_s.unit):
                        matching_abs = abs_s
                        break

            if matching_abs:
                # 역산: prev = current - diff
                derived_prev = matching_abs.value - diff_s.value
                # [수정 v6.23] prev_time_period — '지난해 같은 달/기간'이므로
                # 현재 시점에서 1년 전으로 계산. (이전 버그: 현재 시점을
                # 그대로 넣어 prev==current → fetch가 두 시점 확보 실패 →
                # difference claim이 '단일 fetch로 검증 불가'로 끝남)
                # LLM이 prev_time_period를 채웠으면 그대로 존중, 비었으면 역산.
                _llm_prev_tp = getattr(diff_s, "prev_time_period", None)
                _derived_prev_tp = _llm_prev_tp or _prev_year_period(
                    matching_abs.time_period
                )
                # diff_s에 prev_value 채워넣기 (model_copy)
                try:
                    updated = diff_s.model_copy(update={
                        "prev_value": derived_prev,
                        "prev_time_period": _derived_prev_tp,
                        "prev_phrase": None,  # 역산이라 원문 phrase 없음
                    })
                    # results 안에서 교체
                    for i, s in enumerate(results):
                        if s is diff_s:
                            results[i] = updated
                            break
                    logger.info(
                        f"  ✨ prev_value 역산 (L fix): {diff_s.indicator}={diff_s.value} "
                        f"← {matching_abs.indicator}={matching_abs.value} - {diff_s.value} "
                        f"= {derived_prev:.4f} (prev_time={_derived_prev_tp})"
                    )
                except Exception as e:
                    logger.debug(f"prev_value 역산 실패: {e}")

    return results


# ── [v6.14 E fix] value 환산 정확성 검증 ───────────────────────────────

def _verify_and_correct_value(
    value: float | None,
    source_phrase: str,
) -> tuple[float | None, bool]:
    """
    LLM이 보낸 value의 환산 정확성을 source_phrase로부터 검증.

    "2만 171" → 21710 같은 환산 오류를 잡아냄:
    - source_phrase에서 우리 코드로 수치 추출 (한글 단위 포함)
    - 그 수치 집합 중 value와 *가장 가까운 값* 찾음
    - 차이가 0.5% 이상이면 → 그 가까운 값으로 교체

    Returns:
        (corrected_value, was_corrected_flag)
    """
    if value is None or not source_phrase:
        return value, False

    sp_numbers = _extract_numbers_from_text(source_phrase)
    if not sp_numbers:
        return value, False  # 환산 불가능한 표현 → 그대로

    # value에 가장 가까운 수
    closest = min(sp_numbers, key=lambda n: abs(n - value))

    # 차이 작으면 OK
    if abs(closest - value) < 0.001:
        return value, False
    if value != 0 and abs(closest - value) / abs(value) < 0.005:
        return value, False

    # 차이 큼 — closest로 교체
    return float(closest), True


# ── [v6.14] Context leak 방지를 위한 검증 헬퍼 ───────────────────────────────

def _source_phrase_in_claim(phrase: str, claim_text: str) -> bool:
    """
    LLM이 제공한 source_phrase가 claim_text에 등장하는지 검증.

    [v6.15] 3단계 비교:
      1) 직접 substring
      2) 공백 제거 후 비교
      3) 숫자 기준 비교 — source_phrase의 모든 숫자가 claim_text에 있으면 통과
         ("6.8%" vs "6.8%↑" 처럼 기호 차이로 1·2단계가 실패하는 경우 대응)
    """
    if not phrase or not claim_text:
        return False
    # 1) 직접 substring
    if phrase in claim_text:
        return True
    # 2) 공백 제거 후 비교 (예: "1만 7921 건" vs "1만 7921건")
    phrase_no_space = re.sub(r"\s+", "", phrase)
    claim_no_space = re.sub(r"\s+", "", claim_text)
    if phrase_no_space in claim_no_space:
        return True
    # 3) [v6.15] 숫자 기준 비교 — 기호(↑↓%、 등) 차이 흡수
    #    source_phrase의 숫자들이 모두 claim_text 안에 있으면 leak 아님
    phrase_nums = re.findall(r"\d+\.?\d*", phrase)
    if phrase_nums:
        claim_nums = set(re.findall(r"\d+\.?\d*", claim_text))
        if all(n in claim_nums for n in phrase_nums):
            return True
    return False


def _value_in_claim_text(value: float, claim_text: str) -> bool:
    """
    source_phrase가 누락된 경우 fallback. value의 환산 전 표기가 문장에 있는지 검증.

    예: value=20171 → "2만 171", "20171", "20,171" 등 매칭 시도.
    """
    if value is None:
        return True
    if not claim_text:
        return False

    # 텍스트에서 모든 수치(한글 단위 포함) 추출 → 집합 만들고 value와 매칭
    numbers = _extract_numbers_from_text(claim_text)
    for n in numbers:
        if abs(n - value) < 0.001:
            return True
        if value != 0 and abs(n - value) / abs(value) < 0.005:
            return True
    return False


def _extract_numbers_from_text(text: str) -> set[float]:
    """
    텍스트에서 한글 단위 포함 모든 수치 추출.

    - "2만 171" → 20171
    - "1만 9059" → 19059
    - "6.7" → 6.7
    - "0.76" → 0.76
    - "238,317" → 238317
    """
    numbers: set[float] = set()

    # [박재윤 - 2026-05-18] "N만 N천" 복합 패턴 (앞 패턴보다 먼저 실행 필요)
    # "2869만 3000명" → 28693000
    # "159만 명" 같은 경우와 구분: 뒤 숫자가 1000 단위인 경우
    for m in re.finditer(r"(\d+)\s*만\s*(\d{4})", text):
        n = int(m.group(1)) * 10000 + int(m.group(2))
        numbers.add(float(n))

    # 1) 한글 단위 — "N만 M" 또는 "N만"
    for m in re.finditer(r"(\d+)\s*만\s*(\d+)?", text):
        n = int(m.group(1)) * 10000
        if m.group(2):
            n += int(m.group(2))
        numbers.add(float(n))


    # 2) 한글 단위 — "N억 M" 또는 "N억"
    for m in re.finditer(r"(\d+)\s*억\s*(\d+)?", text):
        n = int(m.group(1)) * 100_000_000
        if m.group(2):
            n += int(m.group(2))
        numbers.add(float(n))

    # 3) 한글 단위 — "N천 M" (앞에 만/억 없을 때만)
    for m in re.finditer(r"(?<![만억\d])(\d+)\s*천\s*(\d+)?", text):
        n = int(m.group(1)) * 1000
        if m.group(2):
            n += int(m.group(2))
        numbers.add(float(n))
    
    # "N만 M천" 복합 패턴
    for m in re.finditer(r"(\d+)\s*만\s*(\d+)\s*천", text):
        n = int(m.group(1)) * 10000 + int(m.group(2)) * 1000
        numbers.add(float(n))

    # 4) 일반 숫자 (콤마 포함 정수 + 소수)
    for m in re.finditer(r"[\d,]+(?:\.\d+)?", text):
        s = m.group().replace(",", "")
        if not s or s in (".",):
            continue
        try:
            numbers.add(float(s))
        except ValueError:
            pass

    return numbers


def _validate_schema(schema: ClaimSchema) -> bool:
    """indicator 없으면 KOSIS 검색 불가 → 실패 처리."""
    if not schema.indicator or len(schema.indicator.strip()) < 2:
        return False
    return True


def _safe_float(v: Any) -> float | None:
    """다양한 수치 표현 → float 변환.

    LLM이 이미 한글 단위를 환산해줘야 하지만, 혹시 문자열로 넘어올 때를 위한 백업.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = re.sub(r"[%,약\s]", "", v.strip())
        match = re.search(r"-?[\d.]+", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None