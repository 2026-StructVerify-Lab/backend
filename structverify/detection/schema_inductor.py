"""
detection/schema_inductor.py — Dynamic Schema Induction (Step 5)

[김예슬 - 2026-04-22]
- SCHEMA_INDUCTION_PROMPT: 도메인 컨텍스트 주입 + 예시 강화
- _safe_float(): 다양한 수치 표현 파싱 ("64.2%", "약 64" 등)
- _validate_schema(): 최소 유효성 검증
- 재시도 로직 추가 (최대 2회)

[김예슬 - 2026-04-24]
- generate_json() → generate_structured() 으로 교체
  · 기존: LLM이 JSON 텍스트 생성 → 직접 파싱 (실패 가능, 재시도 필요)
  · 변경: Structured Outputs (HCX-007) → JSON Schema 보장 (파싱 실패 없음)
- CLAIM_SCHEMA_JSON_SCHEMA: ClaimSchema에 대응하는 JSON Schema 정의 추가
- OpenAI fallback도 response_format으로 동일하게 처리
- 재시도 로직 제거 (Structured Outputs는 파싱 실패 자체가 없음)

[참고] CLOVA Studio Structured Outputs
  https://api.ncloud-docs.com/docs/en/clovastudio-chatcompletionsv3-so
[참고] AutoSchemaKG (arXiv 2505.23628)
"""
from __future__ import annotations

import re
from typing import Any

from structverify.core.schemas import Claim, ClaimSchema
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# 도메인별 indicator 힌트
DOMAIN_HINTS: dict[str, str] = {
    "agriculture": "농가 수, 경작면적, 수확량, 고령화비율, 후계농 비율, 농업소득 등",
    "economy":     "경제성장률, 소비자물가지수, 수출액, 취업자 수, 산업생산지수 등",
    "finance":     "금리, 환율, 주가지수, 대출잔액, 가계부채비율 등",
    "population":  "인구수, 합계출산율, 기대수명, 고령화비율, 인구증가율 등",
    "employment":  "고용률, 실업률, 임금상승률, 취업자 수, 근로시간 등",
    "healthcare":  "의료기관 수, 사망률, 질환자 수, 의료비, 건강보험료 등",
    "education":   "학생 수, 진학률, 교육비, 학교 수, 졸업률 등",
}

# ── Structured Outputs용 JSON Schema 정의 ────────────────────────────────
# HCX-007이 반드시 이 형식으로 반환하도록 강제
CLAIM_SCHEMA_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "indicator": {
            "type": "string",
            "description": "측정하는 핵심 지표 (예: 65세 이상 경영주 비율)"
        },
        "time_period": {
            "type": "string",
            "description": "기준 연도/시점 (예: 2023, 2023년 1분기)"
        },
        "unit": {
            "type": "string",
            "description": "수치 단위 (예: %, 만명, ha, 억원)"
        },
        "population": {
            "type": "string",
            "description": "대상 집단/범위 (예: 과수 농가, 전국, 15~64세)"
        },
        "value": {
            "type": "number",
            "description": "주장에 나온 수치 (숫자만)"
        },
        "source_reference": {
            "type": "string",
            "description": "주장에 언급된 출처 기관/보고서"
        },
        "graph_schema_candidates": {
            "type": "array",
            "description": "Knowledge Graph 노드/엣지 후보",
            "items": {
                "type": "object",
                "properties": {
                    "node_type": {"type": "string"},
                    "label": {"type": "string"},
                    "edge_type": {"type": "string"},
                    "from": {"type": "string"},
                    "to": {"type": "string"}
                }
            },
            "maxItems": 6
        }
    },
    "required": ["indicator", "time_period", "unit", "population"]
}

SCHEMA_INDUCTION_PROMPT = """아래 주장에서 공식 통계 검증에 필요한 핵심 정보를 추출하세요.

주장: "{claim_text}"
도메인: {domain}
{domain_hint}

[추출 기준]
- indicator: 측정하는 핵심 지표
- time_period: 기준 연도/시점
- unit: 수치 단위
- population: 대상 집단/범위
- value: 주장에 나온 수치 (없으면 null)
- source_reference: 언급된 출처 (없으면 null)
- graph_schema_candidates: KG 노드/엣지 후보 (최대 6개)"""


async def induce_schemas(claims: list[Claim], config: dict | None = None) -> list[Claim]:
    """
    각 주장에서 ClaimSchema를 동적으로 유도한다.

    Structured Outputs(HCX-007) 사용으로 JSON 파싱 실패 없음.
    실패 시 재시도 없이 schema=None으로 처리.
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))
    success, fail = 0, 0

    for claim in claims:
        domain = config.get("detected_domain", "general")
        domain_hint = f"주요 지표 예시: {DOMAIN_HINTS[domain]}" if domain in DOMAIN_HINTS else ""

        schema = await _induce_single(llm, claim.claim_text, domain, domain_hint)
        if schema:
            claim.schema = schema
            success += 1
            logger.info(
                f"스키마 유도: {claim.sent_id} "
                f"indicator={schema.indicator}, value={schema.value}, unit={schema.unit}"
            )
        else:
            fail += 1
            logger.warning(f"스키마 유도 실패: {claim.sent_id}")

    logger.info(f"스키마 유도 완료: 성공 {success}건, 실패 {fail}건")
    return claims


async def _induce_single(
    llm: LLMClient,
    claim_text: str,
    domain: str = "general",
    domain_hint: str = "",
) -> ClaimSchema | None:
    """
    단일 주장 → ClaimSchema 변환.

    Structured Outputs 사용:
      HCX: generate_structured() → HCX-007 Structured Outputs
      OpenAI: generate_structured() → response_format json_schema
    """
    prompt = SCHEMA_INDUCTION_PROMPT.format(
        claim_text=claim_text,
        domain=domain,
        domain_hint=domain_hint,
    )

    try:
        r = await llm.generate_structured(
            prompt=prompt,
            schema=CLAIM_SCHEMA_JSON_SCHEMA,
            system_prompt="통계 분석 전문가. 정확한 정보만 추출하세요.",
        )

        schema = ClaimSchema(
            indicator=r.get("indicator") or None,
            time_period=r.get("time_period") or None,
            unit=r.get("unit") or None,
            population=r.get("population") or None,
            value=_safe_float(r.get("value")),
            source_reference=r.get("source_reference") or None,
            graph_schema_candidates=r.get("graph_schema_candidates") or [],
        )

        if not _validate_schema(schema):
            logger.warning(f"스키마 유효성 미달: {claim_text[:50]}")
            return None

        return schema

    except Exception as e:
        logger.warning(f"스키마 유도 예외: {e}")
        return None


def _validate_schema(schema: ClaimSchema) -> bool:
    """indicator 없으면 KOSIS 검색 불가 → 실패 처리"""
    if not schema.indicator or len(schema.indicator.strip()) < 2:
        return False
    return True


def _safe_float(v: Any) -> float | None:
    """다양한 수치 표현 → float 변환"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = re.sub(r"[%,약\s]", "", v.strip())
        match = re.search(r"[\d.]+", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None