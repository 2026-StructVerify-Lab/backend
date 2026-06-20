"""[리팩] agent 프로필 판정 — loop._synthesize_verdict_* 로직 추출 (이동만)"""
from __future__ import annotations

from typing import Any

from structverify.agent.schemas import ClaimType
from structverify.core.schemas import Claim, VerdictType
from structverify.utils.logger import get_logger

from .adapters import AgentCalculateInput, AgentFetchInput, VerdictDecision
from .growth_diff import try_difference_from_rows, try_growth_rate_from_rows
from .verdict_thresholds import (
    classify_atomic_ratio_agent,
    classify_calculate_simple_agent,
    classify_difference_gap_agent,
    classify_growth_rate_pp_agent,
    detect_threshold_direction,
    growth_rate_direction_mismatch,
)

logger = get_logger(__name__)

_COMPLEX_TYPES = {ClaimType.GROWTH_RATE, ClaimType.DIFFERENCE, ClaimType.RANKING}


def decide_verdict_agent_fetch(
    claim: Claim,
    normalized: AgentFetchInput,
    config: dict,
) -> VerdictDecision:
    """fetch observation 기반 agent 판정 (loop._synthesize_verdict_from_observation).

    합성 규칙:
      - fetch 성공 + claim에 값 있음 → 값 비교 (tolerance, 기본 5%)
      - growth_rate/difference/ranking → 두 시점 비교 필요인데 단일 fetch 뿐 → UNVERIFIABLE
        (단, GROWTH_RATE/DIFFERENCE는 rows pool에서 직접 계산 시도)
      - fetch 실패 또는 값 없음 → UNVERIFIABLE
    """
    claim_id = normalized.claim_id
    evidence = normalized.evidence
    tolerance = normalized.tolerance
    claim_actual_type = normalized.claim_actual_type

    fetched_value = evidence.get("value")
    fetched_unit = evidence.get("unit", "") or ""
    fetched_time = evidence.get("time_period", "") or ""
    stat_table_id = evidence.get("stat_table_id", "") or ""
    stat_name = evidence.get("stat_name", "") or ""

    schema = claim.schema
    claim_value = schema.value if schema is not None else None
    claim_unit = (schema.unit or "") if schema is not None else ""
    claim_time = (schema.time_period or "") if schema is not None else ""
    claim_indicator = (schema.indicator or "") if schema is not None else ""

    # 복합 claim type: 두 시점 비교 필요인데 plan은 단일 fetch
    # ★ plan.claim_type은 Planner LLM이 source_text 의미로 일괄 분류해서 부정확함
    #   → claim.schema에서 직접 추론한 type을 더 신뢰
    if isinstance(claim_actual_type, ClaimType) and claim_actual_type in _COMPLEX_TYPES:
        # ── [v6.17] GROWTH_RATE 직접 계산 시도 ──────────────────────────
        if claim_actual_type == ClaimType.GROWTH_RATE:
            result = _try_growth_rate_verdict(
                claim, claim_id, evidence, schema, claim_value, claim_indicator,
                stat_table_id, normalized.all_fetch_observations, config,
            )
            if result is not None:
                return result

        if claim_actual_type == ClaimType.DIFFERENCE:
            result = _try_difference_verdict(
                claim, claim_id, evidence, schema, claim_value,
                stat_table_id, normalized.all_fetch_observations, config,
            )
            if result is not None:
                return result

        logger.info(
            f"[loop] {claim_id}: claim type={claim_actual_type.value} "
            f"(planner type={normalized.plan_claim_type.value}) — 단일 fetch로 검증 불가"
        )
        return VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            explanation=(
                f"{claim_actual_type.value} 유형은 두 시점 비교 필요. "
                f"KOSIS({stat_table_id}) 현재값 {fetched_value!r}{fetched_unit} "
                f"(시점 {fetched_time}) 확보. "
                f"이전 시점 데이터 부재로 검증 불가."
            ),
        )

    if fetched_value is None or claim_value is None:
        return VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            explanation=(
                f"비교 불가: 주장값={claim_value!r}{claim_unit}, "
                f"KOSIS({stat_table_id}) 조회값={fetched_value!r}{fetched_unit}."
            ),
        )

    try:
        fv = float(fetched_value)
        cv = float(claim_value)
    except (TypeError, ValueError):
        return VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            explanation=(
                f"값 숫자 변환 실패 — 주장값={claim_value!r}, 조회값={fetched_value!r}."
            ),
        )

    if abs(cv) < 1e-9:
        diff_ratio = 0.0 if abs(fv) < 1e-9 else 1.0
    else:
        diff_ratio = abs(fv - cv) / abs(cv)

    time_aligned = True
    if claim_time and fetched_time:
        ct_norm = str(claim_time).replace("-", "").replace(".", "")
        ft_norm = str(fetched_time).replace("-", "").replace(".", "")
        time_aligned = (ct_norm in ft_norm) or (ft_norm in ct_norm)

    src_label = f"KOSIS({stat_table_id})" + (f" {stat_name}" if stat_name else "")

    # [v6.20] 부등식 주장 — threshold 방향 감지 후 실측값 vs 기준값 비교
    thr_dir = detect_threshold_direction(claim)
    if thr_dir is not None and time_aligned:
        if thr_dir == "gte":
            satisfied = fv >= cv
            rel_txt = "이상"
        else:
            satisfied = fv <= cv
            rel_txt = "이하"
        logger.info(
            f"[loop] {claim_id}: threshold 판정 dir={thr_dir} "
            f"기준값={cv:.4g} 실측={fv:.4g} → {'충족' if satisfied else '미충족'}"
        )
        if satisfied:
            return VerdictDecision(
                claim_id=claim_id,
                verdict=VerdictType.MATCH,
                confidence=0.8,
                explanation=(
                    f"주장은 '{cv:.4g}{claim_unit} {rel_txt}'(부등식)이고, "
                    f"{src_label} 조회값은 {fv:.4g}{fetched_unit} "
                    f"(시점 {fetched_time or claim_time})이므로 주장이 성립합니다."
                ),
            )
        return VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.MISMATCH,
            confidence=0.7,
            explanation=(
                f"주장은 '{cv:.4g}{claim_unit} {rel_txt}'(부등식)이지만, "
                f"{src_label} 조회값은 {fv:.4g}{fetched_unit} "
                f"(시점 {fetched_time or claim_time})이므로 주장이 성립하지 않습니다."
            ),
        )

    if diff_ratio < tolerance and time_aligned:
        return VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.MATCH,
            confidence=0.85,
            explanation=(
                f"주장값 {cv:.4g}{claim_unit}과 {src_label} 조회값 "
                f"{fv:.4g}{fetched_unit}이 일치 (오차 {diff_ratio*100:.2f}%, "
                f"시점 주장={claim_time or '?'}, 조회={fetched_time or '?'})."
            ),
        )

    if not time_aligned:
        return VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.35,
            explanation=(
                f"시점 불일치 — 주장은 {claim_time}, {src_label} 조회는 {fetched_time}. "
                f"동일 시점 데이터 미확보로 검증 불가 "
                f"(조회값 {fv:.4g}{fetched_unit}, 주장값 {cv:.4g}{claim_unit})."
            ),
        )

    verdict_t, conf = classify_atomic_ratio_agent(diff_ratio, config)
    return VerdictDecision(
        claim_id=claim_id,
        verdict=verdict_t,
        confidence=conf,
        explanation=(
            f"주장값 {cv:.4g}{claim_unit}과 {src_label} 조회값 "
            f"{fv:.4g}{fetched_unit}이 {diff_ratio*100:.1f}% 차이 "
            f"(시점 {fetched_time or claim_time})."
        ),
    )


def decide_verdict_agent_calculate(
    claim: Claim,
    normalized: AgentCalculateInput,
    config: dict,
) -> VerdictDecision:
    """calculate observation 기반 agent 판정 (loop._synthesize_verdict_from_calculate).

    LLM이 prev/current를 계산했지만 finish를 안 부르고 다시 같은 액션 반복
    → 중복차단 → 강제 unverifiable로 죽는 케이스 회복. calculate output의
    result 값을 claim.schema.value와 비교해 자동 verdict 생성한다.
    """
    claim_id = normalized.claim_id
    calc_value = normalized.calc_value
    claim_actual_type = normalized.claim_actual_type
    calc_summary = normalized.calc_summary

    schema = claim.schema
    claim_value = schema.value if schema is not None else None
    claim_unit = (schema.unit or "") if schema is not None else ""
    cv = float(claim_value)

    if isinstance(claim_actual_type, ClaimType) and claim_actual_type == ClaimType.GROWTH_RATE:
        diff = abs(abs(calc_value) - abs(cv))
        verdict_t, conf, label = classify_growth_rate_pp_agent(diff, config)
        diff_desc = f"차이 {diff:.2f}%p"
    elif isinstance(claim_actual_type, ClaimType) and claim_actual_type == ClaimType.DIFFERENCE:
        gap = abs(abs(calc_value) - abs(cv))
        verdict_t, conf = classify_difference_gap_agent(gap, cv, config)
        tol = max(abs(cv) * 0.10, 0.02)
        label = "일치" if verdict_t == VerdictType.MATCH else (
            "오차 큼" if verdict_t == VerdictType.UNVERIFIABLE else "불일치"
        )
        diff_desc = f"차이 {gap:.4f}, 허용 {tol:.4f}"
    else:
        if abs(cv) < 1e-9:
            diff_ratio = 0.0 if abs(calc_value) < 1e-9 else 1.0
        else:
            diff_ratio = abs(calc_value - cv) / abs(cv)
        verdict_t, conf = classify_calculate_simple_agent(diff_ratio, config)
        label = "일치" if verdict_t == VerdictType.MATCH else "불일치"
        diff_desc = f"오차 {diff_ratio*100:.2f}%"

    logger.info(
        f"[loop] {claim_id}: calculate 합성 판정={label} "
        f"(기사 {cv}{claim_unit} vs 계산 {calc_value:.4g}, {diff_desc})"
    )
    return VerdictDecision(
        claim_id=claim_id,
        verdict=verdict_t,
        confidence=conf,
        explanation=(
            f"Agent가 직접 계산한 결과로 검증: 기사 주장 {cv}{claim_unit}, "
            f"산출된 값 {calc_value:.4g} ({diff_desc}). "
            f"계산식: {calc_summary[:200]}"
        ),
    )


def _try_growth_rate_verdict(
    claim: Claim,
    claim_id: str,
    evidence: dict,
    schema: Any,
    claim_value: Any,
    claim_indicator: str,
    stat_table_id: str,
    all_fetch_observations: list | None,
    config: dict,
) -> VerdictDecision | None:
    calc = try_growth_rate_from_rows(
        evidence, schema, claim_id, all_fetch_observations=all_fetch_observations,
    )
    if calc is None or claim_value is None:
        return None
    calc_rate, _cur_v, _prev_v, calc_desc = calc
    try:
        claimed_rate = float(claim_value)
    except (TypeError, ValueError):
        return None

    if growth_rate_direction_mismatch(claim_indicator, calc_rate):
        # [패치 J] 증가/감소 방향 불일치 — 부호 가드
        diff = abs(abs(calc_rate) - abs(claimed_rate))
        logger.warning(
            f"[loop] {claim_id}: growth_rate 부호 방향 불일치 "
            f"(indicator={claim_indicator!r}, 기사 {claimed_rate:+.2f}% 방향, "
            f"계산 {calc_rate:+.2f}% 반대 방향) → MISMATCH 강제"
        )
        return VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.MISMATCH,
            confidence=0.75,
            explanation=(
                f"증가율 방향 불일치: 기사는 '{claim_indicator}' "
                f"{claimed_rate}% (양의 방향), "
                f"KOSIS({stat_table_id}) 표 계산값 "
                f"{calc_rate:.2f}% ({'감소' if calc_rate < 0 else '증가'} 방향). "
                f"{calc_desc}"
            ),
        )

    diff = abs(abs(calc_rate) - abs(claimed_rate))
    verdict_t, conf = classify_growth_rate_pp_agent(diff, config)
    v_label = {
        VerdictType.MATCH: "일치",
        VerdictType.UNVERIFIABLE: "오차 큼",
        VerdictType.MISMATCH: "불일치",
    }[verdict_t]
    logger.info(
        f"[loop] {claim_id}: growth_rate 직접계산 판정={v_label} "
        f"(기사 {claimed_rate}% vs 계산 {calc_rate:.2f}%, 차이 {diff:.2f}%p)"
    )
    return VerdictDecision(
        claim_id=claim_id,
        verdict=verdict_t,
        confidence=conf,
        explanation=(
            f"증가율 직접 검증: 기사 주장 {claimed_rate}%, "
            f"KOSIS({stat_table_id}) 표에서 계산한 값 "
            f"{calc_rate:.2f}% (차이 {diff:.2f}%p). {calc_desc}"
        ),
    )


def _try_difference_verdict(
    claim: Claim,
    claim_id: str,
    evidence: dict,
    schema: Any,
    claim_value: Any,
    stat_table_id: str,
    all_fetch_observations: list | None,
    config: dict,
) -> VerdictDecision | None:
    calc = try_difference_from_rows(
        evidence, schema, claim_id, all_fetch_observations=all_fetch_observations,
    )
    if calc is None or claim_value is None:
        return None
    calc_diff, _cur_v, _prev_v, calc_desc = calc
    try:
        claimed_diff = float(claim_value)
    except (TypeError, ValueError):
        return None

    gap = abs(abs(calc_diff) - abs(claimed_diff))
    verdict_t, conf = classify_difference_gap_agent(gap, claimed_diff, config)
    tol = max(abs(claimed_diff) * 0.10, 0.02)
    v_label = {
        VerdictType.MATCH: "일치",
        VerdictType.UNVERIFIABLE: "오차 큼",
        VerdictType.MISMATCH: "불일치",
    }[verdict_t]
    logger.info(
        f"[loop] {claim_id}: difference 직접계산 판정={v_label} "
        f"(기사 {claimed_diff} vs 계산 {calc_diff:.4f}, "
        f"차이 {gap:.4f}, 허용 {tol:.4f})"
    )
    return VerdictDecision(
        claim_id=claim_id,
        verdict=verdict_t,
        confidence=conf,
        explanation=(
            f"차이값 직접 검증: 기사 주장 {claimed_diff}, "
            f"KOSIS({stat_table_id}) 표에서 계산한 값 "
            f"{calc_diff:.4f} (차이 {gap:.4f}). {calc_desc}"
        ),
    )
