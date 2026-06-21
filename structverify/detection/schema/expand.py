"""detection/schema/expand.py — induce_schemas 후처리 (dedup + claim 복제).

schema_inductor.py에서 분리 (로직 move-only, 동작 변경 없음).

[v6.13] 한 claim → 여러 ClaimSchema 시 claim 복제
[v6.17] value=null 중복 schema 제거
[2026-05-21] seen_keys dedup — agent loop 회귀 차단
"""
from __future__ import annotations

from uuid import uuid4

from structverify.core.schemas import Claim, ClaimSchema
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _dedup_null_schemas(schemas: list[ClaimSchema]) -> list[ClaimSchema]:
    # [v6.17] value=null 중복 schema 제거
    #   LLM이 value를 못 채우고 indicator만 같은 빈 schema를 N개 만드는
    #   경우만 정리. 단, population까지 같아야 진짜 중복으로 간주.
    #   ★ "동작구 10.6%, 성동구 8.9%"처럼 지역만 다른 정상 다중 수치는
    #     population이 다르므로 합쳐지지 않음 (이전엔 다 뭉개지던 버그).
    # [2026-05-21] seen_keys로 통합 — 같은 (indicator, time, population) 키에
    #   *value 있는* schema가 이미 존재하면 *value=null* 후속 schema는 폐기.
    #   효과: LLM이 base claim에 "합계출산율 0.79명" 정상 schema +
    #   "합계출산율 null" 빈 schema를 *함께* 출력하던 케이스에서, 빈 schema가
    #   별도 sub-claim으로 살아남아 agent loop이 4 iter 돌다가
    #   "주장값=None명 vs KOSIS 0.8명" 으로 끝나던 회귀를 차단.
    deduped: list[ClaimSchema] = []
    seen_keys: set[tuple] = set()
    for sch in schemas:
        key = (
            sch.indicator or "",
            sch.time_period or "",
            sch.population or "",   # ★ population 추가 — 지역별 구분
        )
        if sch.value is None and key in seen_keys:
            logger.info(
                f"  [중복 제거] value=null schema 폐기 — 같은 키의 "
                f"value 있는 schema가 이미 존재 "
                f"(indicator={sch.indicator}, time={sch.time_period}, "
                f"population={sch.population})"
            )
            continue
        seen_keys.add(key)
        deduped.append(sch)
    return deduped


def _expand_claims_from_schemas(
    claim: Claim,
    schemas: list[ClaimSchema],
) -> list[Claim]:
    """첫 schema는 원래 claim에 부착, 나머지는 claim 복제 후 부착."""
    if not schemas:
        return []

    # 첫 schema는 원래 claim에 부착
    claim.schema = schemas[0]
    expanded: list[Claim] = [claim]
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
        logger.info(
            f"스키마 유도: {claim.sent_id} [{i}/{len(schemas)}] (복제) "
            f"indicator={sch.indicator}, value={sch.value}, "
            f"unit={sch.unit}, time_period={sch.time_period}, "
            f"parent_path={sch.parent_path}"
        )
    return expanded
