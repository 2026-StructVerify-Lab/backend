"""
detection/schema_inductor.py — Dynamic Schema Induction (Step 5)

기존 고정 필드 추출(Schema Extraction)에서 LLM이 도메인에 맞는 스키마를
동적으로 유도하는 Schema Induction으로 확장.

[김예슬 - 2026-04-22]
- SCHEMA_INDUCTION_PROMPT: 도메인 컨텍스트 주입 + 예시 강화
  · 도메인별 indicator / population 힌트 제공
  · graph_schema_candidates 노드/엣지 타입 가이드 구체화
- induce_schemas(): 도메인 정보(sir_doc.detected_domain)를 프롬프트에 주입
- _safe_float(): 문자열 수치 파싱 강화 ("64.2%", "약 64" 등 처리)
- _validate_schema(): 유도된 스키마 최소 유효성 검증
- 재시도 로직 추가 (최대 2회)

[참고] AutoSchemaKG (arXiv 2505.23628)
  LLM이 텍스트에서 그래프 스키마(엔티티 타입, 관계 타입)를 자동 유도.
  본 프로젝트의 graph_schema_candidates 필드 설계에 직접 참고.

[참고] ProgramFC (Pan et al., NAACL 2023)
  복잡한 주장을 indicator/time/unit/population으로 분해하는 구조 참고.
"""
from __future__ import annotations

import re
from typing import Any

from structverify.core.schemas import Claim, ClaimSchema
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# 도메인별 indicator 힌트 — 프롬프트에 주입하여 LLM 유도 정확도 향상
DOMAIN_HINTS: dict[str, str] = {
    "agriculture": "농가 수, 경작면적, 수확량, 고령화비율, 후계농 비율, 농업소득 등",
    "economy": "경제성장률, 소비자물가지수, 수출액, 취업자 수, 산업생산지수 등",
    "finance": "금리, 환율, 주가지수, 대출잔액, 가계부채비율 등",
    "population": "인구수, 합계출산율, 기대수명, 고령화비율, 인구증가율 등",
    "employment": "고용률, 실업률, 임금상승률, 취업자 수, 근로시간 등",
    "healthcare": "의료기관 수, 사망률, 질환자 수, 의료비, 건강보험료 등",
    "education": "학생 수, 진학률, 교육비, 학교 수, 졸업률 등",
}

SCHEMA_INDUCTION_PROMPT = """통계 분석 전문가로서 아래 주장에서 검증에 필요한 핵심 정보를 추출하고,
Knowledge Graph 연결을 위한 스키마 후보도 함께 유도하세요.

주장: "{claim_text}"
도메인: {domain}
{domain_hint}

[추출 기준]
- indicator: 측정하는 핵심 지표 (예: "65세 이상 고령화 비율", "합계출산율")
- time_period: 기준 연도/시점 (예: "2023", "2023년 1분기", "지난해")
- unit: 수치 단위 (예: "%", "만명", "ha", "억원")
- population: 대상 집단/범위 (예: "과수 농가", "전국", "15~64세")
- value: 주장에 나온 수치 (숫자만, 없으면 null)
- source_reference: 주장에 언급된 출처 기관/보고서 (없으면 null)
- graph_schema_candidates: Knowledge Graph 노드/엣지 후보

[예시]
주장: "2023년 기준 65세 이상 과수 농가 경영주 비율이 64.2%에 달했다"
→ {{
  "indicator": "65세 이상 경영주 비율",
  "time_period": "2023",
  "unit": "%",
  "population": "과수 농가",
  "value": 64.2,
  "source_reference": null,
  "graph_schema_candidates": [
    {{"node_type": "metric", "label": "65세이상경영주비율"}},
    {{"node_type": "entity", "label": "과수농가"}},
    {{"node_type": "time", "label": "2023"}},
    {{"edge_type": "measured_at", "from": "65세이상경영주비율", "to": "2023"}},
    {{"edge_type": "belongs_to", "from": "65세이상경영주비율", "to": "과수농가"}}
  ]
}}

JSON으로만 답하세요:
{{
  "indicator": "...",
  "time_period": "...",
  "unit": "...",
  "population": "...",
  "value": 숫자 또는 null,
  "source_reference": "..." 또는 null,
  "graph_schema_candidates": [...]
}}"""


async def induce_schemas(claims: list[Claim], config: dict | None = None) -> list[Claim]:
    """
    각 주장에서 Schema를 동적으로 유도한다. (AutoSchemaKG 방식)

    LLM이 도메인에 맞는 indicator/time_period/unit/population/value를 추출하고,
    Knowledge Graph 연결을 위한 노드/엣지 후보(graph_schema_candidates)도 함께 생성한다.

    Args:
        claims: Claim 리스트 (claim.schema는 None 상태)
        config: 설정 dict (llm, domain 설정 포함)

    Returns:
        schema가 채워진 Claim 리스트
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    success, fail = 0, 0

    for claim in claims:
        # 도메인 힌트 — 해당 도메인의 주요 지표 정보를 LLM에 제공
        domain = config.get("detected_domain", "general")
        domain_hint_str = ""
        if domain in DOMAIN_HINTS:
            domain_hint_str = f"주요 지표 예시: {DOMAIN_HINTS[domain]}"

        schema = await _induce_single(llm, claim.claim_text, domain, domain_hint_str)

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
    max_retry: int = 2,
) -> ClaimSchema | None:
    """
    단일 주장에 대해 스키마를 유도한다.
    실패 시 max_retry 횟수만큼 재시도한다.
    """
    prompt = SCHEMA_INDUCTION_PROMPT.format(
        claim_text=claim_text,
        domain=domain,
        domain_hint=domain_hint,
    )

    for attempt in range(1, max_retry + 1):
        try:
            r = await llm.generate_json(
                prompt=prompt,
                system_prompt="통계 분석 전문가. JSON으로만 답하세요.",
                model_tier="heavy",  # HCX-003 — 정밀 추출이 필요
            )

            # JSON 파싱 실패 케이스
            if "raw" in r:
                logger.warning(f"스키마 JSON 파싱 실패 (시도 {attempt}/{max_retry}): {claim_text[:50]}")
                continue

            schema = ClaimSchema(
                indicator=r.get("indicator") or None,
                time_period=r.get("time_period") or None,
                unit=r.get("unit") or None,
                population=r.get("population") or None,
                value=_safe_float(r.get("value")),
                source_reference=r.get("source_reference") or None,
                graph_schema_candidates=r.get("graph_schema_candidates") or [],
            )

            # 최소 유효성 검증 — indicator 없으면 재시도
            if not _validate_schema(schema):
                logger.warning(f"스키마 유효성 미달 (시도 {attempt}/{max_retry}): {claim_text[:50]}")
                continue

            return schema

        except Exception as e:
            logger.warning(f"스키마 유도 예외 (시도 {attempt}/{max_retry}): {e}")

    return None


def _validate_schema(schema: ClaimSchema) -> bool:
    """
    유도된 스키마의 최소 유효성 검증.
    indicator가 없으면 검색 쿼리를 만들 수 없으므로 실패 처리.
    """
    if not schema.indicator or len(schema.indicator.strip()) < 2:
        return False
    return True


def _safe_float(v: Any) -> float | None:
    """
    다양한 형태의 수치 표현을 float으로 변환.

    처리 케이스:
    - None → None
    - 64.2 → 64.2
    - "64.2" → 64.2
    - "64.2%" → 64.2
    - "약 64" → 64.0
    - "6만4천" → None (복잡한 한국어 수치는 포기)
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        # % 기호, 공백, "약" 등 제거
        cleaned = re.sub(r"[%,약\s]", "", v.strip())
        # 숫자와 소수점만 남기기
        match = re.search(r"[\d.]+", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None