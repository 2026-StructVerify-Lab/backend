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
    "required": ["indicator", "time_period", "unit", "population", "parent_path"],
}


# ── List wrapper Schema — 한 문장 → N개 schema ─────────────────────
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

검증 대상 문장: "{claim_text}"
문맥 (참고용): {context}
도메인: {domain}
{domain_hint}
{temporal_hint}

[작업 목표]
이 문장에 *검증 가능한 수치 주장이 여러 개* 있을 수 있습니다.
각 수치 주장마다 *별도 schema* 객체로 추출하세요.
결과는 schemas 배열로 반환합니다.

[예시 — 한 문장에 2개 수치]
문장: "올해 4월 출생아 수는 2만 171명으로 지난해보다 6.7% 늘었다"
결과:
  schemas: [
    {{indicator: "출생아 수", value: 20171, unit: "명", time_period: "2025-04",
      parent_path: "인구 > 출생 > 출생아 수"}},
    {{indicator: "출생아 수", value: 6.7, unit: "%", time_period: "2025-04",
      parent_path: "인구 > 출생 > 출생아 수"}}
  ]
  (절대값 schema + 증가율 schema 둘 다. indicator는 같지만 value/unit이 다름)

[예시 — 한 문장에 1개 수치]
문장: "올해 합계출산율은 0.76명이다"
결과:
  schemas: [
    {{indicator: "합계출산율", value: 0.76, unit: "명", time_period: "2025",
      parent_path: "인구 > 출생 > 합계출산율"}}
  ]

[예시 — rank 표현 (검증 불가)]
문장: "출생아 수, 34년 만에 최대 증가"
결과:
  schemas: []   (절대 수치 없음, 순위 표현만 → 검증 불가)

[예시 — 비교 기준값 + 변화량]
문장: "합계출산율 0.76명으로 지난해보다 0.04명 증가"
결과:
  schemas: [
    {{indicator: "합계출산율", value: 0.76, unit: "명", time_period: "2025"}},
    {{indicator: "합계출산율 차이", value: 0.04, unit: "명", time_period: "2025"}}
  ]

[핵심 규칙]

1. **단위 통일**: 한글 단위는 정확하게 숫자로 변환. *'만'은 10,000*.
   · "2만 171명"  → value=20171 (NOT 2,171,000)
   · "1만 9059명" → value=19059
   · "21만 7천명" → value=217000
   · "1만 7921건" → value=17921

2. **★ 한 문장에 여러 수치**: 각각 별도 schema. *놓치지 마세요*.
   · "X명으로 Y% 늘었다" → 2개 schema:
     - 절대값: indicator=X측정대상, value=X, unit=명
     - 비율:   indicator=X측정대상, value=Y, unit=%
   · "X로 Z 증가" → 2개 schema:
     - 현재값: value=X
     - 변화량: indicator="X측정대상 차이", value=Z

3. **★ value와 unit 일관성**: unit="%" 면 value는 비율(-100~100),
   unit="명/건/원" 이면 value는 절대값.

4. **단위 강제**: unit 절대 비우지 마세요. 불분명하면: 지수/배/위/점.

5. **indicator는 측정 대상 자체**:
   ✗ "출생아 수 증가율"  →  ○ "출생아 수" (unit="%"로 구분)
   단, *차이/변화량*은 별도 indicator OK ("합계출산율 차이").

6. **분류 축은 population**:
   ✗ indicator="대졸이상 쉬었음 청년"
   ○ indicator="쉬었음 인구", population="대졸 이상 청년"

7. **parent_path**: "대분류 > 중분류 > 소분류". 기사 제목/출처/기관명 금지.
   KOSIS 대분류: 인구/가구/고용/노동/임금/물가/가계/보건/사회/복지/교육/
   환경/농림/수산/건설/주택/토지/교통/정보통신/경제/산업/무역.

8. **시점 형식**: "YYYY" 또는 "YYYY-MM"만. "2024년 4월" → "2024-04".

9. **순위 주장 단독**: "N년 만에 최대" 단독이면 schemas=[] 반환.
   문장에 다른 측정 숫자가 있으면 그것만 추출.

10. **근사 표현**: "안팎/이상/약/가량" 있으면 is_approximate=true, modifier=원문.

11. **value 출처**: 반드시 *검증 대상 문장*에 있는 숫자만. 문맥에만 있는
    숫자는 사용 금지.
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
    Structured Outputs 사용 — JSON 파싱 실패 없음.
    실패 시 빈 리스트.
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
                "한 문장에 여러 수치 주장 있으면 *모두* 별도 schema로. "
                "indicator/unit/parent_path 누락 금지."
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
        try:
            schema = ClaimSchema(
                indicator=item.get("indicator") or None,
                time_period=item.get("time_period") or None,
                unit=item.get("unit") or None,
                population=item.get("population") or None,
                value=_safe_float(item.get("value")),
                source_reference=item.get("source_reference") or None,
                graph_schema_candidates=item.get("graph_schema_candidates") or [],
                parent_path=item.get("parent_path") or None,
                is_approximate=bool(item.get("is_approximate", False)),
                modifier=item.get("modifier") or None,
            )
        except Exception as e:
            logger.debug(f"개별 schema 파싱 실패: {e}")
            continue

        if _validate_schema(schema):
            results.append(schema)

    return results


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