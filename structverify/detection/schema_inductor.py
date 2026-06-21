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

from structverify.core.schemas import Claim
from structverify.detection.prompts.schema import DOMAIN_HINTS

from structverify.detection.schema.validate import _source_phrase_in_claim

from structverify.detection.schema.expand import (
    _dedup_null_schemas,
    _expand_claims_from_schemas,
)
from structverify.detection.schema.induce import _induce_multiple
from structverify.detection.schema.regenerate import regenerate_schema
from structverify.detection.schema.temporal_hints import _build_temporal_hint
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


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

        schemas = _dedup_null_schemas(schemas)
        new_claims = _expand_claims_from_schemas(claim, schemas)
        expanded.extend(new_claims)
        success += len(new_claims)

    logger.info(
        f"스키마 유도 완료: {len(claims)}개 claim → {len(expanded)}개 claim "
        f"(성공 schema {success}건, 실패 claim {fail}건)"
    )
    return expanded
