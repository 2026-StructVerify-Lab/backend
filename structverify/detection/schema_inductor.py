"""
detection/schema_inductor.py — Dynamic Schema Induction (Step 5)

기존 고정 필드 추출(Schema Extraction)에서 LLM이 도메인에 맞는 스키마를
동적으로 유도하는 Schema Induction으로 확장.

[참고] AutoSchemaKG (arXiv 2505.23628) — https://github.com/NousResearch/AutoSchemaKG
  LLM이 텍스트에서 그래프 스키마(엔티티 타입, 관계 타입)를 자동 유도하는 방법론.
  본 프로젝트의 graph_schema_candidates 필드 설계에 직접 참고.

[참고] ProgramFC (Pan et al., NAACL 2023) — https://github.com/mbzuai-nlp/ProgramFC
  복잡한 주장을 indicator/time/unit/population으로 분해하는 구조 참고.
"""
from __future__ import annotations
from structverify.core.schemas import Claim, ClaimSchema
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_INDUCTION_PROMPT = """통계 분석 전문가로서 아래 주장에서 검증에 필요한 핵심 정보를 추출하고,
그래프 스키마 후보도 함께 유도하세요.

주장: "{claim_text}"

JSON:
{{
  "indicator": "측정 지표명",
  "time_period": "기준 시점",
  "unit": "단위",
  "population": "대상 범위",
  "value": 수치(숫자만),
  "source_reference": "언급된 출처 또는 null",
  "graph_schema_candidates": [
    {{"node_type": "entity|metric|time", "label": "노드 라벨"}},
    {{"edge_type": "measured_at|belongs_to", "from": "시작노드", "to": "끝노드"}}
  ]
}}"""


async def induce_schemas(claims: list[Claim], config: dict | None = None) -> list[Claim]:
    """
    각 주장에서 Schema를 동적으로 유도 (AutoSchemaKG 방식).
    LLM이 도메인에 맞는 노드/엣지 타입까지 함께 유도한다.
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))
    for claim in claims:
        try:
            r = await llm.generate_json(
                SCHEMA_INDUCTION_PROMPT.format(claim_text=claim.claim_text))
            claim.schema = ClaimSchema(
                indicator=r.get("indicator"), time_period=r.get("time_period"),
                unit=r.get("unit"), population=r.get("population"),
                value=_safe_float(r.get("value")),
                source_reference=r.get("source_reference"),
                graph_schema_candidates=r.get("graph_schema_candidates", []),
            )
            logger.info(f"스키마 유도 완료: {claim.sent_id}")
        except Exception as e:
            logger.error(f"스키마 유도 실패 [{claim.sent_id}]: {e}")
    return claims


def _safe_float(v) -> float | None:
    if v is None: return None
    try: return float(v)
    except (ValueError, TypeError): return None
