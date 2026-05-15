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
                "prev_value의 시점. 형식: 'YYYY' 또는 'YYYY-MM'. "
                "예: 현재가 2025-04이고 '지난해 같은 달'이면 '2024-04'. "
                "추론 가능하면 채우고, 불명확하면 빈 문자열."
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

[핵심 규칙]

1. **단위 통일**: 한글 단위는 정확하게 숫자로 변환. *'만'은 10,000*.
   · "2만 171명"  → value=20171 (NOT 2,171,000)
   · "1만 9059명" → value=19059
   · "21만 7천명" → value=217000
   · "1만 7921건" → value=17921

2. **★ 한 문장에 여러 수치**: 각각 별도 schema. *놓치지 마세요*.
   · "X명으로 Y% 늘었다" → 2개 schema (절대값 + 비율)
   · "X로 Z 증가" → 2개 schema (현재값 + 변화량)

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

        schemas = await _induce_multiple(
            llm, claim.claim_text, domain, domain_hint,
            context=context, temporal_hint=temporal_hint,
        )

        if not schemas:
            # 검증 가능 수치 0개 — 원래 claim은 유지하되 schema=None
            fail += 1
            expanded.append(claim)
            logger.warning(
                f"스키마 유도: {claim.sent_id} → 검증 가능 수치 없음"
            )
            continue

        # 첫 schema는 원래 claim에 부착
        claim.schema = schemas[0]
        expanded.append(claim)
        success += 1
        logger.info(
            f"스키마 유도: {claim.sent_id} [1/{len(schemas)}] "
            f"indicator={schemas[0].indicator}, value={schemas[0].value}, "
            f"unit={schemas[0].unit}, parent_path={schemas[0].parent_path}"
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
                f"unit={sch.unit}, parent_path={sch.parent_path}"
            )

    logger.info(
        f"스키마 유도 완료: {len(claims)}개 claim → {len(expanded)}개 claim "
        f"(성공 schema {success}건, 실패 claim {fail}건)"
    )
    return expanded


def _build_temporal_hint(graph: "ClaimGraph", claim: Claim) -> str:
    """그래프 시점 해소 결과를 prompt hint 텍스트로."""
    prov = graph.temporal_provenance(claim)
    anchor_year = graph.get_anchor_year()

    if prov and prov.get("resolved"):
        return (
            f"\n[시점 정보 — 그래프 해소 결과]\n"
            f"- 원문 표현: {prov.get('expression')}\n"
            f"- 해소된 절대 시점: {prov['resolved']}\n"
            f"- 근거: {prov.get('basis') or '문서 anchor 기반'}\n"
            f"위 절대 시점을 time_period로 사용하세요."
        )
    elif anchor_year is not None:
        return (
            f"\n[시점 정보 — 문서 anchor]\n"
            f"- 이 문서의 기준 연도(anchor_year): {anchor_year}\n"
            f"- 본문에 '작년/지난해/재작년/올해' 같은 상대 표현이 있으면\n"
            f"  anchor_year를 기준으로 절대 연도(예: {anchor_year-1}, {anchor_year-2})로 풀어\n"
            f"  time_period에 절대값으로 적으세요."
        )
    return ""


async def _induce_multiple(
    llm: LLMClient,
    claim_text: str,
    domain: str = "general",
    domain_hint: str = "",
    context: str = "",
    temporal_hint: str = "",
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
            if prev_phrase and not _source_phrase_in_claim(prev_phrase, claim_text):
                logger.warning(
                    f"  ⚠️ prev_phrase context leak: {prev_phrase!r} 가 검증 대상 문장에 없음 "
                    f"→ prev_value 폐기 (indicator={item.get('indicator')})"
                )
                prev_value = None
                prev_time_period = None
                prev_phrase = None

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

            schema_kwargs = dict(
                indicator=item.get("indicator") or None,
                time_period=item.get("time_period") or None,
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

            schema = ClaimSchema(**schema_kwargs)
        except Exception as e:
            logger.debug(f"개별 schema 파싱 실패: {e}")
            continue

        if _validate_schema(schema):
            results.append(schema)

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

    엄격 비교 (substring) + 공백 무관 비교 둘 다 시도.
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
