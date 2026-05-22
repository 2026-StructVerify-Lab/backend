"""Document-level Dependency Planner — Phase D+ (2026-05-21).

각 sub-claim을 *독립적으로* planner+agent_loop에 보내던 기존 흐름의 한계:
  - 같은 문장에서 분기된 base + derived sub-claim이 동시에 병렬 실행됨
  - derived(예: 출생아 수 증가율)는 base(예: 출생아 수)의 KOSIS evidence가 있으면
    재활용 가능한데, 병렬이라 base 결과가 아직 없어 derived가 또 fetch 시도
  - 같은 indicator를 공유하는 base 그룹(의료장비 4개 시도 케이스)도 catalog 결과를
    공유하지 못해 각자 catalog_search를 4회 반복

처방: sub-claim들을 *실행 레벨*로 묶어 level 간 *순차*, level 내 *병렬*로 처리.
verified_facts/successful_stat_ids 캐시가 level 간에 살아 있으므로 level 2의 claim들이
level 1의 evidence를 자동 재활용.

레벨 결정 신호:
  - schema.value_role
    - "base"               → Level 1 (단독 검증)
    - "aggregation"        → Level 1 (다년 fetch — sibling 의존 없음, catalog 캐시 공유)
    - "derived_rate"       → Level 2 (base 결과 의존)
    - "derived_difference" → Level 2
    - None / 기타          → Level 1 (default)

옵션 (use_llm=True): LLM이 모든 sub-claim의 schema를 보고 더 정밀한 dependency 그래프
구성. 현재는 stub (deterministic으로 fallback). Phase C+에서 prompt 구현 예정.
"""
from __future__ import annotations

from typing import Any, Callable, Awaitable

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 공개 API ──────────────────────────────────────────────────────────


def build_execution_levels(
    claims: list[Any],
    *,
    llm_call: Callable[[str], Awaitable[str]] | None = None,
    use_llm: bool = False,
) -> list[list[Any]]:
    """sub-claim 리스트 → 실행 레벨 (list of lists).

    Args:
        claims: structverify Claim 리스트
        llm_call: 옵션. LLM 기반 dependency 분석 (use_llm=True일 때 사용)
        use_llm: True면 LLM document_planner 호출, False면 결정론 분석

    Returns:
        levels: [[claim, ...], [claim, ...], ...]
            - 외부 list는 순차 실행 (level 간 순서 의존)
            - 내부 list는 병렬 실행 가능 (level 안의 claim들)
        모든 입력 claim은 정확히 한 level에 포함됨.
    """
    if not claims:
        return []

    if use_llm and llm_call is not None:
        try:
            levels = _llm_levels(claims, llm_call)
            if levels and _all_claims_covered(claims, levels):
                logger.info(
                    f"[dep_planner] LLM mode: {len(levels)} levels, "
                    f"sizes={[len(lvl) for lvl in levels]}"
                )
                return levels
            logger.warning(
                "[dep_planner] LLM 응답 부적합 → 결정론으로 fallback"
            )
        except Exception as e:
            logger.warning(
                f"[dep_planner] LLM mode 실패 → 결정론으로 fallback: "
                f"{type(e).__name__}: {e}"
            )

    levels = _deterministic_levels(claims)
    logger.info(
        f"[dep_planner] deterministic mode: {len(levels)} levels, "
        f"sizes={[len(lvl) for lvl in levels]}"
    )
    return levels


# ── 결정론 분석 ──────────────────────────────────────────────────────


def _deterministic_levels(claims: list[Any]) -> list[list[Any]]:
    """결정론적 level 분할 — schema.value_role 기반.

    Level 1: base + value_role 미지정 claim
        - 단독 검증 가능. catalog_search → fetch → finish.
        - 같은 indicator를 공유하는 claim들은 prior_success stat_id 캐시로
          catalog 결과 자연스럽게 공유 (현재 매커니즘 그대로).

    Level 2: derived_rate / derived_difference claim
        - Level 1 끝난 후 실행. base claim의 fetch evidence가 verified_facts에
          캐시되어 있어 prev/current 시점 fetch를 재활용할 수 있음.
    """
    base_or_unspecified: list[Any] = []
    derived: list[Any] = []

    for c in claims:
        schema = getattr(c, "schema", None)
        role = getattr(schema, "value_role", None) if schema else None
        if role in ("derived_rate", "derived_difference"):
            derived.append(c)
        else:
            # "base" / None / 기타 → Level 1
            base_or_unspecified.append(c)

    levels: list[list[Any]] = []
    if base_or_unspecified:
        levels.append(base_or_unspecified)
    if derived:
        levels.append(derived)

    # 안전망: 모든 claim이 한 level에 들어갔는지
    if not _all_claims_covered(claims, levels):
        logger.warning(
            "[dep_planner] 결정론 분할에서 누락 claim 발견 → 단일 level fallback"
        )
        return [list(claims)]

    return levels


def _all_claims_covered(claims: list[Any], levels: list[list[Any]]) -> bool:
    """levels에 모든 claim이 정확히 한 번씩 들어갔는지 검증."""
    flat = [c for lvl in levels for c in lvl]
    if len(flat) != len(claims):
        return False
    flat_ids = {id(c) for c in flat}
    return all(id(c) in flat_ids for c in claims)


# ── LLM 기반 분석 (stub — Phase C+) ──────────────────────────────────


def _llm_levels(
    claims: list[Any],
    llm_call: Callable[[str], Awaitable[str]],
) -> list[list[Any]]:
    """LLM이 모든 sub-claim의 schema를 보고 dependency 그래프 구성.

    현재는 stub. Phase C+에서 prompt 구현 예정. 호출자는 fallback을 처리해야 함.
    """
    raise NotImplementedError(
        "LLM document_planner는 Phase C+에서 구현 예정. "
        "현재는 use_llm=False (결정론)만 작동."
    )
