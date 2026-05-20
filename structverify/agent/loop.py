"""structverify.agent.loop — Agent Loop (Phase D).

Pipeline:
  Plan (Phase C) → **Loop** → AgentVerdict

Loop 책임:
  1. Plan의 initial_steps을 *순서대로 실행* (deterministic mode)
  2. 각 step의 결과를 Observation으로 wrap + memory/log 기록
  3. FINISH action 또는 max_iter 도달 시 종료
  4. **Reflect hook** — 매 iteration 전에 *결정 함수* 호출 가능 (Phase E에서 진짜 Reflect Agent)
  5. **★ Auto verdict synthesis** — plan steps 소진 시 마지막 fetch observation 기반
     deterministic verdict 자동 합성 (Phase E에서 LLM verdict로 대체)

이번 Phase D = **deterministic mode만**. Plan의 initial_steps 그대로 실행.
Phase E에서 reflect_fn으로 *LLM이 다음 step 결정* 가능.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from structverify.utils.logger import get_logger
from .schemas import (
    ActionType,
    AgentVerdict,
    ClaimType,
    DataPointSpec,
    Observation,
    Plan,
    PlanStep,
    ReflectDecision,
    StopReason,
    VerdictType,
)
from .tools import get_tool_class, ToolContext, ToolResult
from .memory import append_iteration, append_plan_summary
from .workspace import Workspace

logger = get_logger(__name__)

# ── Reflect Hook (Phase E에서 LLM 기반으로 대체 가능) ────────────

class ReflectFn(Protocol):
    """매 iteration 전에 *다음 행동 결정* 함수.

    Args:
        plan: 현재 plan
        memory_text: 지금까지의 memory.md 내용 (LLM이 본 컨텍스트)
        last_observation: 직전 iteration 결과 (None이면 첫 iter)
        iter_num: 현재 iteration 번호 (1-based)

    Returns:
        ReflectDecision (다음 action + input + rationale)
        또는 None (Loop이 기본 동작 — plan의 다음 step 그대로 실행)
    """

    async def __call__(
        self,
        plan: Plan,
        memory_text: str,
        last_observation: Observation | None,
        iter_num: int,
    ) -> ReflectDecision | None:
        ...


# ── Loop 설정 ────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    """Loop 동작 설정."""

    max_iterations: int = 10
    """최대 iteration 수. 도달 시 강제 unverifiable 종료."""

    mode: str = "deterministic"
    """'deterministic' (plan 그대로) | 'reflect' (reflect_fn 호출, Phase E)."""

    fail_fast: bool = False
    """True면 첫 Tool 실패 시 즉시 unverifiable. False면 계속 다음 step 시도."""

    value_match_tolerance: float = 0.01
    """auto verdict 합성 시 값 매칭 허용 오차 (1% 기본)."""


# ── Step input 보간 (deterministic mode 보조) ────────────────────

def _interpolate_step_input(
    step: PlanStep,
    last_observation: Observation | None,
) -> PlanStep:
    """deterministic mode에서 step.input의 placeholder를 직전 observation 결과로 치환.

    Planner가 plan 만들 때 *catalog_search 결과를 아직 모르므로* fetch_evidence의
    candidate_id에 placeholder 문자열을 넣음 (예: '<catalog_search 결과의 top id>').
    Loop이 deterministic mode에서 그걸 그대로 넘기면 fetch 실패 → 여기서 보간.
    """
    if step.action != ActionType.FETCH_EVIDENCE:
        return step
    if last_observation is None or last_observation.action != ActionType.CATALOG_SEARCH:
        return step

    cid = (step.input or {}).get("candidate_id", "")
    is_placeholder = (
        not cid
        or (isinstance(cid, str) and (
            cid.startswith("<")
            or cid.startswith("{")
            or cid.strip().upper() in {"TBD", "TODO", "FILL_ME", "N/A"}
        ))
    )
    if not is_placeholder:
        return step

    candidates = (last_observation.output or {}).get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        logger.warning(
            f"[loop] 보간 skip: last_observation.output에 candidates 없음 "
            f"(action={last_observation.action.value}, output keys={list((last_observation.output or {}).keys())})"
        )
        return step
    top_id = candidates[0].get("id")
    if not top_id:
        return step

    new_input = dict(step.input or {})
    new_input["candidate_id"] = top_id
    # ── [v6.18] 후보 순회용 fallback 리스트 ──────────────────────────
    # catalog top 1개가 무관한 표일 수 있으므로(예: "연평균기온"에
    # "[해양기상] 등표 관측값"이 1등), 나머지 후보 id들도 같이 넘겨서
    # fetch_evidence가 관련성 체크 실패 시 다음 후보로 재시도하게 함.
    fallback_ids = [
        c.get("id") for c in candidates[1:]
        if isinstance(c, dict) and c.get("id")
    ]
    if fallback_ids:
        new_input["_candidate_fallbacks"] = fallback_ids
    logger.info(
        f"[loop] candidate_id placeholder 보간: {cid!r} → {top_id!r} "
        f"(fallback 후보 {len(fallback_ids)}개)"
    )
    return PlanStep(action=step.action, input=new_input, rationale=step.rationale)


# ── Claim type 추론 (Planner LLM 분류 보정) ──────────────────────

# growth_rate 표현 키워드 (indicator에 들어있으면 growth_rate로 추론)
_GROWTH_INDICATOR_KEYWORDS = ("증가율", "증감률", "증감율", "성장률", "비율", "퍼센트", "%")
# difference 표현 키워드
_DIFF_INDICATOR_KEYWORDS = ("차이", "증감", "감소", "증가분", "감소분", "변화량", "격차")
# ranking 표현 키워드
_RANK_INDICATOR_KEYWORDS = ("순위", "1위", "최고", "최대", "최저", "최소", "가장 높")
# growth_rate 단위
_GROWTH_UNITS = ("%", "%p", "퍼센트", "%P", "pp")

# [v6.20] threshold(부등식) 표현 키워드.
# "14도를 넘기다/돌파하다" 같은 주장은 value=14를 *등호*로 비교하면
# 실측 14.5와 안 맞아 가짜 mismatch가 난다. 사실은 14.5 >= 14 → 충족.
# claim 문장에 아래 표현이 있으면 등호가 아니라 부등식으로 판정한다.
_THRESHOLD_GTE_KEYWORDS = (  # 실측 >= value 면 충족 (이상/초과/돌파)
    "넘기", "넘어", "넘는", "넘은", "넘었", "돌파", "초과",
    "이상", "웃돌", "상회", "넘게",
)
_THRESHOLD_LTE_KEYWORDS = (  # 실측 <= value 면 충족 (이하/미만)
    "미만", "이하", "밑돌", "하회", "못 미", "못미", "안 되", "안되",
)


def _infer_claim_type(claim: Any) -> ClaimType | None:
    """Claim의 *실제* 유형을 schema에서 추론.

    Planner LLM이 source_text 전체 의미로 일괄 분류하기 때문에,
    같은 문장에서 추출된 absolute / growth_rate claim들이 모두 growth_rate로
    뭉뚱그려지는 문제가 있음. claim.schema의 indicator/unit/prev_value를 보면
    정확하게 알 수 있으므로 그것으로 보정.

    우선순위 (v6.17 수정 — prev_value를 unit보다 먼저 체크):
      1. ClaimSchema.comparison_type (명시되어 있으면)
      2. Claim.canonical_type
      3. indicator 키워드 매칭 (순위/차이 → ranking/difference)
      4. prev_value 있음 + unit % → growth_rate (두 시점 비교 비율)
      5. prev_value 있음 (unit % 아님) → comparison/difference
      6. prev_value 없음 → unit 무관하게 ABSOLUTE
         · "공시가격 변동률 6.8%" 처럼 unit이 %여도 비교 기준값이
           문장에 없으면 단일 시점 절대값(ABSOLUTE)으로 봐야 함.
           이전엔 unit=% 만으로 growth_rate라 단정 → planner의
           ranking/comparison 판정과 충돌 → 전부 unverifiable 되던 버그.
    """
    schema = getattr(claim, "schema", None)
    if schema is None:
        return None

    # 1. schema.comparison_type 명시
    comp = getattr(schema, "comparison_type", None)
    if isinstance(comp, ClaimType):
        return comp

    # 2. claim.canonical_type
    canon = getattr(claim, "canonical_type", None)
    if isinstance(canon, ClaimType):
        return canon

    indicator = (getattr(schema, "indicator", None) or "").strip()
    unit = (getattr(schema, "unit", None) or "").strip()
    prev_value = getattr(schema, "prev_value", None)

    # 3. indicator 키워드 — 순위/차이는 unit과 무관하게 먼저 판정
    if any(kw in indicator for kw in _RANK_INDICATOR_KEYWORDS):
        return ClaimType.RANKING
    if any(kw in indicator for kw in _DIFF_INDICATOR_KEYWORDS):
        return ClaimType.DIFFERENCE
    if any(kw in indicator for kw in _GROWTH_INDICATOR_KEYWORDS):
        # 증가율/성장률 키워드 → growth_rate (비교 기준 필요)
        return ClaimType.GROWTH_RATE

    # 4. prev_value 있음 → 두 시점 비교 claim
    if prev_value is not None:
        if unit in _GROWTH_UNITS:
            return ClaimType.GROWTH_RATE
        return ClaimType.COMPARISON

    # 5. prev_value 없음 → 단일 시점 절대값
    #    unit이 %여도 비교 기준이 없으면 growth_rate가 아님.
    return ClaimType.ABSOLUTE


# ── Auto verdict synthesis (Phase D deterministic) ───────────────

def _evidence_to_data_points(evidence: dict, claim: Any) -> list[DataPointSpec]:
    """fetch observation의 evidence dict → DataPointSpec 리스트.

    [v6.17] agent 경로가 검증에 쓴 KOSIS 데이터를 verdict에 담아야
    runtime_agent가 그걸 VerificationResult.evidence로 복원해서 UI에 표시함.
    이전엔 data_points=[] 로 비워 → UI에 '공식 통계 출처'가 안 떴음.
    """
    if not evidence:
        return []
    fetched_value = evidence.get("value")
    if fetched_value is None:
        # 값 없으면 출처 표시 무의미 — 빈 리스트
        return []
    schema = getattr(claim, "schema", None)
    indicator = (getattr(schema, "indicator", "") or "") if schema else ""
    population = (getattr(schema, "population", None)) if schema else None
    stat_id = evidence.get("stat_table_id", "") or ""
    try:
        rv = float(fetched_value)
    except (TypeError, ValueError):
        rv = None
    return [
        DataPointSpec(
            indicator=indicator or (evidence.get("stat_name", "") or "KOSIS"),
            time=str(evidence.get("time_period", "") or ""),
            population=population,
            unit_hint=evidence.get("unit", "") or None,
            resolved_value=rv,
            resolved_unit=evidence.get("unit", "") or None,
            source=(f"KOSIS:{stat_id}" if stat_id else "KOSIS"),
            source_time=str(evidence.get("time_period", "") or "") or None,
        )
    ]


def _save_verified_facts(
    workspace: Any, verdict: Any, claim_id: str
) -> None:
    """[v6.21] verdict의 data_points에서 검증된 수치를 job 공유 저장소에 기록.

    MATCH/MISMATCH verdict는 KOSIS 공식 수치를 data_points에 담는다.
    그 (indicator, time_period, value, unit)을 verified_facts에 저장하면,
    다음 claim이 같은 수치를 catalog_search 없이 재사용할 수 있다.

    UNVERIFIABLE은 공식 수치가 없으므로 저장하지 않는다.
    """
    try:
        v_type = getattr(verdict.verdict, "value", str(verdict.verdict))
        if v_type not in ("match", "mismatch"):
            return  # 검증 실패 — 신뢰할 수치 없음
        dps = getattr(verdict, "data_points", None) or []
        for dp in dps:
            val = getattr(dp, "resolved_value", None)
            if val is None:
                continue
            workspace.append_verified_fact({
                "indicator": getattr(dp, "indicator", "") or "",
                "time_period": (getattr(dp, "source_time", None)
                                or getattr(dp, "time", "") or ""),
                "value": val,
                "unit": getattr(dp, "resolved_unit", None) or "",
                "source": getattr(dp, "source", None) or "KOSIS",
                "claim_id": str(claim_id),
                "verdict": v_type,
            })
    except Exception as e:
        logger.debug(f"[loop] verified_fact 저장 실패 (무시): {e}")


def _find_row_value_for_time(rows: list, target_time: str) -> float | None:
    """[v6.17] KOSIS 표 rows에서 특정 시점(PRD_DE) 행의 값(DT)을 찾는다.

    growth_rate 직접 계산용 — 같은 표에서 1년 전(prev) 값을 추출한다.
    target_time: 'YYYY' 또는 'YYYY-MM'. PRD_DE는 'YYYY' 또는 'YYYYMM' 형식.
    """
    if not rows or not target_time:
        return None
    # 'YYYY-MM' → 'YYYYMM' 정규화
    norm = str(target_time).replace("-", "").strip()
    for row in rows:
        if not isinstance(row, dict):
            continue
        prd = str(row.get("PRD_DE", "") or "").strip()
        if prd == norm:
            v = _parse_row_dt(row.get("DT"))
            if v is not None:
                return v
    # 연도만으로 재시도 (target이 'YYYY-MM'인데 표는 연 단위인 경우)
    year = norm[:4]
    if year and year != norm:
        for row in rows:
            if not isinstance(row, dict):
                continue
            prd = str(row.get("PRD_DE", "") or "").strip()
            if prd == year:
                v = _parse_row_dt(row.get("DT"))
                if v is not None:
                    return v
    return None


def _parse_row_dt(raw: Any) -> float | None:
    """KOSIS row의 DT 필드를 float로 파싱 (콤마/공백 제거)."""
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _try_growth_rate_from_rows(
    evidence: dict,
    schema: Any,
    claim_id: str,
) -> tuple[float, float, float, str] | None:
    """[v6.17] growth_rate claim을 같은 표 rows로 직접 계산.

    KOSIS에 '증가율' 통계표가 따로 없을 때, fetch한 표(현재값 표)의 rows에서
    prev_time_period 시점의 행을 찾아 (current-prev)/prev*100 을 직접 계산한다.
    추가 API 호출 불필요 — evidence dict에 rows 전체가 들어있음.

    반환: (계산된_증가율, current_value, prev_value, 설명) 또는 None.
    """
    prev_time = getattr(schema, "prev_time_period", None) if schema else None
    if not prev_time:
        return None  # 비교 시점 없음 → 계산 불가

    rows = evidence.get("rows") or []
    if not rows:
        return None

    # current 값: fetch가 매칭한 값 (best_row 우선, 없으면 evidence value)
    current_val = _parse_row_dt(evidence.get("value"))
    matched_row = evidence.get("matched_row") or {}
    if current_val is None and matched_row:
        current_val = _parse_row_dt(matched_row.get("DT"))
    if current_val is None:
        return None

    # prev 값: 같은 표에서 prev_time 행 탐색
    prev_val = _find_row_value_for_time(rows, prev_time)
    if prev_val is None:
        logger.info(
            f"[loop] {claim_id}: growth_rate 직접계산 — prev 시점 {prev_time!r} "
            f"행을 표에서 못 찾음 ({len(rows)} rows)"
        )
        return None
    if prev_val == 0:
        return None  # 0으로 나눗셈 방지

    calc_rate = (current_val - prev_val) / prev_val * 100.0
    desc = (
        f"표에서 직접 계산: 현재값 {current_val} - 이전값({prev_time}) {prev_val} "
        f"→ 증가율 ({current_val}-{prev_val})/{prev_val}×100 = {calc_rate:.2f}%"
    )
    logger.info(f"[loop] {claim_id}: growth_rate 직접계산 성공 — {desc}")
    return (calc_rate, current_val, prev_val, desc)


def _try_difference_from_rows(
    evidence: dict,
    schema: Any,
    claim_id: str,
) -> tuple[float, float, float, str] | None:
    """[수정 v6.23] difference claim을 같은 표 rows로 직접 계산.

    growth_rate와 동일한 원리 — fetch한 표의 rows에서 prev_time_period
    시점 행을 찾아 (current - prev) 차이를 직접 계산한다. growth_rate는
    비율(%)이고 difference는 차이값이라는 점만 다르다.

    "합계출산율 0.79명으로 지난해 같은 달보다 0.06명 증가" 같은 claim 대응:
    같은 표에서 2025-04(0.79)와 2024-04 행을 찾아 차이를 계산.

    반환: (계산된_차이, current_value, prev_value, 설명) 또는 None.
    """
    prev_time = getattr(schema, "prev_time_period", None) if schema else None
    if not prev_time:
        return None  # 비교 시점 없음 → 계산 불가

    rows = evidence.get("rows") or []
    if not rows:
        return None

    # current 값: fetch가 매칭한 값
    current_val = _parse_row_dt(evidence.get("value"))
    matched_row = evidence.get("matched_row") or {}
    if current_val is None and matched_row:
        current_val = _parse_row_dt(matched_row.get("DT"))
    if current_val is None:
        return None

    # prev 값: 같은 표에서 prev_time 행 탐색
    prev_val = _find_row_value_for_time(rows, prev_time)
    if prev_val is None:
        logger.info(
            f"[loop] {claim_id}: difference 직접계산 — prev 시점 {prev_time!r} "
            f"행을 표에서 못 찾음 ({len(rows)} rows)"
        )
        return None

    calc_diff = current_val - prev_val
    desc = (
        f"표에서 직접 계산: 현재값({evidence.get('time_period') or '현재'}) "
        f"{current_val} - 이전값({prev_time}) {prev_val} "
        f"→ 차이 {current_val}-{prev_val} = {calc_diff:.4f}"
    )
    logger.info(f"[loop] {claim_id}: difference 직접계산 성공 — {desc}")
    return (calc_diff, current_val, prev_val, desc)


def _detect_threshold_direction(claim: Any) -> str | None:
    """[v6.20] claim 문장이 부등식(threshold) 주장인지 판정.

    Returns:
        "gte"  — "14도를 넘기다/돌파/이상" → 실측 >= value 면 충족
        "lte"  — "14도 미만/이하/밑돌다"  → 실측 <= value 면 충족
        None   — 부등식 표현 없음 (일반 등호 비교)

    claim_text와 schema.modifier 양쪽을 본다. schema_inductor가
    "이상" 등을 modifier로 떼어내므로 거기도 확인.
    """
    text = (getattr(claim, "claim_text", "") or "")
    schema = getattr(claim, "schema", None)
    modifier = ""
    if schema is not None:
        modifier = (getattr(schema, "modifier", "") or "")
    haystack = f"{text} {modifier}"

    has_gte = any(kw in haystack for kw in _THRESHOLD_GTE_KEYWORDS)
    has_lte = any(kw in haystack for kw in _THRESHOLD_LTE_KEYWORDS)

    # 양쪽 다 잡히면 모호 → 등호 비교로 폴백 (안전)
    if has_gte and has_lte:
        return None
    if has_gte:
        return "gte"
    if has_lte:
        return "lte"
    return None


def _synthesize_verdict_from_observation(
    plan: Plan,
    claim: Any,
    claim_id: str,
    last_observation: Observation | None,
    iter_num: int,
    tolerance: float,
) -> AgentVerdict | None:
    """Plan steps 소진 시 fetch observation 보고 deterministic verdict 합성.

    Phase D의 임시 verdict 결정 로직. Phase E에서 LLM 기반 verdict로 교체 예정.

    합성 규칙:
      - last_observation이 fetch_evidence 성공 + claim에 값 있음 → 값 비교 (1% 오차)
        · 일치 → MATCH (conf 0.85)
        · 불일치 → MISMATCH (conf 0.7)
      - growth_rate/difference/ranking → 두 시점 비교 필요인데 단일 fetch 뿐 → UNVERIFIABLE
      - fetch 실패 또는 값 없음 → UNVERIFIABLE
      - last_observation 없거나 fetch 아니면 → None (호출자가 default unverifiable)
    """
    if last_observation is None:
        return None
    if last_observation.action != ActionType.FETCH_EVIDENCE:
        return None

    # fetch 실패 케이스
    if not last_observation.success:
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.25,
            explanation=f"데이터 조회 실패: {(last_observation.summary or '')[:200]}",
            data_points=[],
            iterations_used=iter_num,
            stop_reason=StopReason.COMPLETED,
        )

    evidence = (last_observation.output or {}).get("evidence") or {}
    fetched_value = evidence.get("value")
    fetched_unit = evidence.get("unit", "") or ""
    fetched_time = evidence.get("time_period", "") or ""
    stat_table_id = evidence.get("stat_table_id", "") or ""
    stat_name = evidence.get("stat_name", "") or ""

    schema = getattr(claim, "schema", None)
    claim_value = getattr(schema, "value", None) if schema is not None else None
    claim_unit = getattr(schema, "unit", "") or "" if schema is not None else ""
    claim_time = getattr(schema, "time_period", "") or "" if schema is not None else ""
    claim_indicator = getattr(schema, "indicator", "") or "" if schema is not None else ""

    # 복합 claim type: 두 시점 비교 필요인데 plan은 단일 fetch
    # ★ plan.claim_type은 Planner LLM이 source_text 의미로 일괄 분류해서 부정확함
    #   → claim.schema에서 직접 추론한 type을 더 신뢰
    complex_types = {ClaimType.GROWTH_RATE, ClaimType.DIFFERENCE, ClaimType.RANKING}
    claim_actual_type = _infer_claim_type(claim) or plan.claim_type
    if isinstance(claim_actual_type, ClaimType) and claim_actual_type in complex_types:
        # ── [v6.17] GROWTH_RATE 직접 계산 시도 ──────────────────────────
        # KOSIS에 '증가율' 통계표가 따로 없어도, fetch한 표(현재값 표)의
        # rows에서 prev_time_period 시점 행을 찾아 증가율을 직접 계산한다.
        # "출생아 수 23만 명으로 1년 전보다 7.7% 줄었다" 같은 claim 대응.
        if claim_actual_type == ClaimType.GROWTH_RATE:
            calc = _try_growth_rate_from_rows(evidence, schema, claim_id)
            if calc is not None and claim_value is not None:
                calc_rate, cur_v, prev_v, calc_desc = calc
                try:
                    claimed_rate = float(claim_value)
                except (TypeError, ValueError):
                    claimed_rate = None
                if claimed_rate is not None:
                    # 증가율 비교 — 부호 무시하고 절대 크기 비교
                    # (기사 "7.7% 줄었다"=감소를 7.7로 표기, 계산값은 -7.69)
                    diff = abs(abs(calc_rate) - abs(claimed_rate))
                    # 증가율은 %p 차이로 판정 (1.5%p 이내 일치)
                    if diff <= 1.5:
                        verdict_t = VerdictType.MATCH
                        conf = 0.8
                        v_label = "일치"
                    elif diff <= 5.0:
                        verdict_t = VerdictType.UNVERIFIABLE
                        conf = 0.4
                        v_label = "오차 큼"
                    else:
                        verdict_t = VerdictType.MISMATCH
                        conf = 0.7
                        v_label = "불일치"
                    logger.info(
                        f"[loop] {claim_id}: growth_rate 직접계산 판정={v_label} "
                        f"(기사 {claimed_rate}% vs 계산 {calc_rate:.2f}%, "
                        f"차이 {diff:.2f}%p)"
                    )
                    return AgentVerdict(
                        claim_id=claim_id,
                        verdict=verdict_t,
                        confidence=conf,
                        explanation=(
                            f"증가율 직접 검증: 기사 주장 {claimed_rate}%, "
                            f"KOSIS({stat_table_id}) 표에서 계산한 값 "
                            f"{calc_rate:.2f}% (차이 {diff:.2f}%p). {calc_desc}"
                        ),
                        data_points=_evidence_to_data_points(evidence, claim),
                        iterations_used=iter_num,
                        stop_reason=StopReason.COMPLETED,
                    )

        # ── [수정 v6.23] DIFFERENCE 직접 계산 시도 ──────────────────────
        # growth_rate와 동일 — 같은 표 rows에서 prev_time_period 행을 찾아
        # (current - prev) 차이를 직접 계산. "합계출산율 0.79명으로 지난해
        # 같은 달보다 0.06명 증가" 같은 claim 대응.
        if claim_actual_type == ClaimType.DIFFERENCE:
            calc = _try_difference_from_rows(evidence, schema, claim_id)
            if calc is not None and claim_value is not None:
                calc_diff, cur_v, prev_v, calc_desc = calc
                try:
                    claimed_diff = float(claim_value)
                except (TypeError, ValueError):
                    claimed_diff = None
                if claimed_diff is not None:
                    # 차이값 비교 — 부호 무시하고 절대 크기 비교
                    # (기사 "0.06명 증가"=0.06, 계산값도 +0.06 방향)
                    gap = abs(abs(calc_diff) - abs(claimed_diff))
                    # 차이값 자체의 크기에 비례한 허용 오차 (10%) 또는
                    # 최소 절대 허용치 중 큰 쪽 — 작은 값(0.06 등)도 견딤.
                    tol = max(abs(claimed_diff) * 0.10, 0.02)
                    if gap <= tol:
                        verdict_t = VerdictType.MATCH
                        conf = 0.8
                        v_label = "일치"
                    elif gap <= tol * 3:
                        verdict_t = VerdictType.UNVERIFIABLE
                        conf = 0.4
                        v_label = "오차 큼"
                    else:
                        verdict_t = VerdictType.MISMATCH
                        conf = 0.7
                        v_label = "불일치"
                    logger.info(
                        f"[loop] {claim_id}: difference 직접계산 판정={v_label} "
                        f"(기사 {claimed_diff} vs 계산 {calc_diff:.4f}, "
                        f"차이 {gap:.4f}, 허용 {tol:.4f})"
                    )
                    return AgentVerdict(
                        claim_id=claim_id,
                        verdict=verdict_t,
                        confidence=conf,
                        explanation=(
                            f"차이값 직접 검증: 기사 주장 {claimed_diff}, "
                            f"KOSIS({stat_table_id}) 표에서 계산한 값 "
                            f"{calc_diff:.4f} (차이 {gap:.4f}). {calc_desc}"
                        ),
                        data_points=_evidence_to_data_points(evidence, claim),
                        iterations_used=iter_num,
                        stop_reason=StopReason.COMPLETED,
                    )

        # 직접 계산 불가 (prev 시점 없음 / 표에 prev 행 없음) → 검증 불가
        logger.info(
            f"[loop] {claim_id}: claim type={claim_actual_type.value} "
            f"(planner type={plan.claim_type.value}) — 단일 fetch로 검증 불가"
        )
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            explanation=(
                f"{claim_actual_type.value} 유형은 두 시점 비교 필요. "
                f"KOSIS({stat_table_id}) 현재값 {fetched_value!r}{fetched_unit} "
                f"(시점 {fetched_time}) 확보. "
                f"이전 시점 데이터 부재로 검증 불가."
            ),
            data_points=_evidence_to_data_points(evidence, claim),
            iterations_used=iter_num,
            stop_reason=StopReason.COMPLETED,
        )

    # 값 둘 다 있어야 비교 가능
    if fetched_value is None or claim_value is None:
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            explanation=(
                f"비교 불가: 주장값={claim_value!r}{claim_unit}, "
                f"KOSIS({stat_table_id}) 조회값={fetched_value!r}{fetched_unit}."
            ),
            data_points=_evidence_to_data_points(evidence, claim),
            iterations_used=iter_num,
            stop_reason=StopReason.COMPLETED,
        )

    # 숫자 변환
    try:
        fv = float(fetched_value)
        cv = float(claim_value)
    except (TypeError, ValueError):
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            explanation=(
                f"값 숫자 변환 실패 — 주장값={claim_value!r}, 조회값={fetched_value!r}."
            ),
            data_points=_evidence_to_data_points(evidence, claim),
            iterations_used=iter_num,
            stop_reason=StopReason.COMPLETED,
        )

    # 오차율 계산
    if abs(cv) < 1e-9:
        diff_ratio = 0.0 if abs(fv) < 1e-9 else 1.0
    else:
        diff_ratio = abs(fv - cv) / abs(cv)

    # 시점 일치 여부 (단순 substring 매칭)
    time_aligned = True
    if claim_time and fetched_time:
        ct_norm = str(claim_time).replace("-", "").replace(".", "")
        ft_norm = str(fetched_time).replace("-", "").replace(".", "")
        # claim "2025-04" → "202504", fetched "202504" 또는 "2025"
        # claim이 월별이고 fetched가 연간이면 비매칭
        time_aligned = (ct_norm in ft_norm) or (ft_norm in ct_norm)

    src_label = f"KOSIS({stat_table_id})" + (f" {stat_name}" if stat_name else "")

    # [v6.20] threshold(부등식) 주장 처리 — "14도를 넘기다/돌파" 등.
    # value를 등호로 비교하면 (14 vs 실측 14.5) 가짜 mismatch가 난다.
    # 부등식이 충족되면 MATCH, 어긋나면 MISMATCH로 판정.
    # 시점이 안 맞으면 아래 일반 로직과 동일하게 unverifiable이 맞으므로
    # time_aligned일 때만 부등식 판정을 적용한다.
    _thr_dir = _detect_threshold_direction(claim)
    if _thr_dir is not None and time_aligned:
        if _thr_dir == "gte":
            satisfied = fv >= cv
            rel_txt = "이상"
        else:  # lte
            satisfied = fv <= cv
            rel_txt = "이하"
        logger.info(
            f"[loop] {claim_id}: threshold 판정 dir={_thr_dir} "
            f"기준값={cv:.4g} 실측={fv:.4g} → {'충족' if satisfied else '미충족'}"
        )
        if satisfied:
            return AgentVerdict(
                claim_id=claim_id,
                verdict=VerdictType.MATCH,
                confidence=0.8,
                explanation=(
                    f"주장은 '{cv:.4g}{claim_unit} {rel_txt}'(부등식)이고, "
                    f"{src_label} 조회값은 {fv:.4g}{fetched_unit} "
                    f"(시점 {fetched_time or claim_time})이므로 주장이 성립합니다."
                ),
                data_points=_evidence_to_data_points(evidence, claim),
                iterations_used=iter_num,
                stop_reason=StopReason.COMPLETED,
            )
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.MISMATCH,
            confidence=0.7,
            explanation=(
                f"주장은 '{cv:.4g}{claim_unit} {rel_txt}'(부등식)이지만, "
                f"{src_label} 조회값은 {fv:.4g}{fetched_unit} "
                f"(시점 {fetched_time or claim_time})이므로 주장이 성립하지 않습니다."
            ),
            data_points=_evidence_to_data_points(evidence, claim),
            iterations_used=iter_num,
            stop_reason=StopReason.COMPLETED,
        )

    if diff_ratio < tolerance and time_aligned:
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.MATCH,
            confidence=0.85,
            explanation=(
                f"주장값 {cv:.4g}{claim_unit}과 {src_label} 조회값 "
                f"{fv:.4g}{fetched_unit}이 일치 (오차 {diff_ratio*100:.2f}%, "
                f"시점 주장={claim_time or '?'}, 조회={fetched_time or '?'})."
            ),
            data_points=_evidence_to_data_points(evidence, claim),
            iterations_used=iter_num,
            stop_reason=StopReason.COMPLETED,
        )

    # 시점 불일치인 경우 mismatch보다 unverifiable이 안전 (잘못된 row일 가능성)
    if not time_aligned:
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.35,
            explanation=(
                f"시점 불일치 — 주장은 {claim_time}, {src_label} 조회는 {fetched_time}. "
                f"동일 시점 데이터 미확보로 검증 불가 "
                f"(조회값 {fv:.4g}{fetched_unit}, 주장값 {cv:.4g}{claim_unit})."
            ),
            data_points=_evidence_to_data_points(evidence, claim),
            iterations_used=iter_num,
            stop_reason=StopReason.COMPLETED,
        )

    # 시점은 맞는데 값이 다름 → mismatch
    return AgentVerdict(
        claim_id=claim_id,
        verdict=VerdictType.MISMATCH,
        confidence=0.7,
        explanation=(
            f"주장값 {cv:.4g}{claim_unit}과 {src_label} 조회값 "
            f"{fv:.4g}{fetched_unit}이 {diff_ratio*100:.1f}% 차이 "
            f"(시점 {fetched_time or claim_time})."
        ),
        data_points=_evidence_to_data_points(evidence, claim),
        iterations_used=iter_num,
        stop_reason=StopReason.COMPLETED,
    )


# ── Agent Loop 본체 ──────────────────────────────────────────────

async def agent_loop(
    plan: Plan,
    claim: Any,
    workspace: Workspace,
    datasources: dict[str, Any],
    config: dict[str, Any] | None = None,
    reflect_fn: ReflectFn | None = None,
    loop_config: LoopConfig | None = None,
) -> AgentVerdict:
    """Agent Loop 실행.

    Args:
        plan: Phase C에서 만든 Plan
        claim: structverify Claim (logging + claim_id + schema용)
        workspace: 이 job의 workspace
        datasources: {name: BaseDataSource} 등록된 source들
        config: 전체 config dict (Tool들이 사용)
        reflect_fn: 옵션. Phase E에서 LLM 기반 Reflect Agent.
                    None이면 *deterministic mode* (plan.initial_steps 그대로 실행)
        loop_config: max_iter, mode 등

    Returns:
        AgentVerdict — workspace에도 자동 저장됨 (FinishTool 호출 시 또는 auto-synthesize 시)
    """
    loop_config = loop_config or LoopConfig()
    config = config or {}
    claim_id = str(getattr(claim, "claim_id", "") or getattr(claim, "id", "unknown"))

    # ── 초기화 ──
    logger.info(
        f"[loop] {claim_id}: 시작. plan.type={plan.claim_type.value}, "
        f"steps={len(plan.initial_steps)}, mode={loop_config.mode}, "
        f"max_iter={loop_config.max_iterations}"
    )

    # plan summary를 memory에 기록
    try:
        plan_summary = (
            f"Plan type: {plan.claim_type.value}\n"
            f"Required data points: {len(plan.required_data)}\n"
            f"Formula: {plan.calculation_formula}\n"
            f"Initial steps: {[s.action.value for s in plan.initial_steps]}\n"
            f"Fallback keywords: {plan.fallback.alternative_keywords}"
        )
        append_plan_summary(workspace, claim_id, plan_summary)
    except Exception as e:
        logger.debug(f"[loop] plan summary memory 기록 실패: {e}")

    # ── 실행 루프 ──
    last_observation: Observation | None = None
    # B2 sanity check용 — finish 이후의 verdict 검증에 쓰임.
    # last_observation은 finish 자체로 덮어쓰여 사라지므로 별도 추적.
    last_fetch_observation: Observation | None = None
    last_result: ToolResult | None = None
    finished = False
    stop_reason = StopReason.MAX_ITERATIONS
    plan_step_idx = 0
    plan_exhausted = False

    # ── [중복 action 차단] reflect 모드 전용 ─────────────────────────
    # reflect(HCX)가 thought엔 "다른 검색어"라 쓰면서 action.input.query는
    # 동일하게 두는 일이 잦다 → 같은 catalog_search를 max_iter까지 반복하는
    # 헛돌이 발생. loop이 결정적으로 차단한다:
    #   - 같은 (action, input) 조합이 이미 실행됐으면 tool 실행을 스킵하고
    #     reflect에게 "이미 시도함, 다른 검색어를 쓰라"는 observation을 줌.
    #   - 연속 중복이 임계치를 넘으면 헛돌이로 보고 loop 종료.
    _seen_action_keys: set[str] = set()
    _consecutive_dup = 0
    _DUP_LIMIT = 2  # 연속 중복 이 횟수 도달 시 종료

    def _action_key(step: PlanStep) -> str:
        """action + 입력으로 중복 판별 키. 문자열 값은 공백 제거 정규화."""
        inp = step.input or {}
        norm = {
            k: (str(v).strip().replace(" ", "") if isinstance(v, str) else v)
            for k, v in inp.items()
        }
        return (
            f"{step.action.value}::"
            f"{json.dumps(norm, sort_keys=True, ensure_ascii=False)}"
        )

    for iter_num in range(1, loop_config.max_iterations + 1):
        # ── 다음 step 결정 ──
        next_step: PlanStep | None = None

        if reflect_fn is not None and loop_config.mode == "reflect":
            # Phase E: LLM Reflect Agent
            try:
                memory_text = workspace.read_memory(claim_id)
                decision = await reflect_fn(plan, memory_text, last_observation, iter_num)
            except Exception as e:
                logger.warning(f"[loop] reflect_fn 실패: {e}, deterministic fallback")
                decision = None

            if decision is not None:
                next_step = PlanStep(
                    action=decision.action,
                    input=decision.input,
                    # ReflectDecision은 'thought' 필드를 씀 (rationale 아님).
                    # PlanStep.rationale에 thought를 매핑.
                    rationale=decision.thought,
                )

        # reflect_fn 없거나 None 반환 → plan의 다음 step 사용
        if next_step is None:
            if plan_step_idx < len(plan.initial_steps):
                next_step = plan.initial_steps[plan_step_idx]
                plan_step_idx += 1
                # Planner가 넣은 placeholder를 직전 observation 결과로 보간
                next_step = _interpolate_step_input(next_step, last_observation)
            else:
                # plan steps 다 썼는데 finish 안 함 → auto verdict 합성 시도
                logger.info(
                    f"[loop] {claim_id}: plan steps 모두 소진, iter {iter_num}에서 종료"
                )
                plan_exhausted = True
                stop_reason = StopReason.MAX_ITERATIONS
                break

        # ── [중복 action 차단] reflect 모드에서만 ──────────────────
        # 같은 (action, input)이 이미 실행됐으면 tool 실행 스킵.
        # reflect에게 "이미 시도함" observation을 줘 다른 결정을 유도.
        if loop_config.mode == "reflect":
            _akey = _action_key(next_step)
            if _akey in _seen_action_keys:
                _consecutive_dup += 1
                logger.warning(
                    f"[loop] {claim_id} iter {iter_num}: 중복 action 감지 "
                    f"(action={next_step.action.value}, 연속 {_consecutive_dup}회) "
                    f"— tool 실행 스킵"
                )
                last_observation = Observation(
                    iter_num=iter_num,
                    action=next_step.action,
                    input=next_step.input,
                    output={},
                    summary=(
                        f"[중복 차단] 이 action({next_step.action.value})과 "
                        f"동일한 입력은 이미 이전 iter에서 시도했고 결과가 같았습니다. "
                        f"같은 검색을 반복하지 마세요 — 반드시 *다른 검색어*를 쓰거나 "
                        f"다른 action(fetch_evidence/read_original/finish)을 선택하세요."
                    ),
                    success=False,
                    error="duplicate_action",
                )
                try:
                    append_iteration(
                        workspace, claim_id,
                        iteration_num=last_observation.iter_num,
                        action=getattr(last_observation.action, "value",
                                        str(last_observation.action)),
                        action_input=last_observation.input,
                        observation_summary=last_observation.summary,
                        success=last_observation.success,
                    )
                except Exception as e:
                    logger.warning(f"[loop] 중복 observation 기록 실패: {e}")
                if _consecutive_dup >= _DUP_LIMIT:
                    logger.warning(
                        f"[loop] {claim_id}: 중복 action {_consecutive_dup}회 연속 "
                        f"→ 헛돌이로 판단, iter {iter_num}에서 종료"
                    )
                    plan_exhausted = True
                    stop_reason = StopReason.MAX_ITERATIONS
                    break
                continue  # 다음 iter — reflect가 다른 결정을 하도록
            # 중복 아님 — 키 등록, 연속 카운터 리셋
            _seen_action_keys.add(_akey)
            _consecutive_dup = 0

        # ── Tool 실행 ──
        logger.info(
            f"[loop] {claim_id} iter {iter_num}: action={next_step.action.value} "
            f"rationale={next_step.rationale!r}"
        )

        ctx = ToolContext(
            workspace=workspace,
            claim_id=claim_id,
            config=config,
            datasources=datasources,
            iter_num=iter_num,
            claim=claim,   # fetch_evidence가 claim.schema 사용
        )

        try:
            tool_cls = get_tool_class(next_step.action)
            tool = tool_cls()
        except KeyError as e:
            logger.warning(f"[loop] Tool 미등록: {next_step.action.value}, skip")
            last_result = ToolResult(
                output={}, summary=f"Tool not registered: {next_step.action.value}",
                success=False, error=str(e),
            )
        else:
            # 입력 검증
            valid, err_msg = tool.validate_input(next_step.input)
            if not valid:
                logger.warning(f"[loop] Tool 입력 검증 실패: {err_msg}")
                last_result = ToolResult(
                    output={}, summary=f"입력 검증 실패: {err_msg}",
                    success=False, error=err_msg,
                )
            else:
                try:
                    last_result = await tool.execute(next_step.input, ctx)
                except Exception as e:
                    logger.exception(f"[loop] Tool 실행 예외: {next_step.action.value}")
                    last_result = ToolResult(
                        output={}, summary=f"실행 예외: {type(e).__name__}: {e}",
                        success=False, error=f"{type(e).__name__}: {e}",
                    )

        # ── Observation 생성 + memory 기록 ──
        last_observation = Observation(
            iter_num=iter_num,
            action=next_step.action,
            input=next_step.input,
            output=last_result.output,
            summary=last_result.summary,
            success=last_result.success,
            error=last_result.error,
        )
        # B2 sanity check용 — fetch가 성공할 때마다 갱신 (finish가 덮어쓰지 못하게)
        if next_step.action == ActionType.FETCH_EVIDENCE and last_result.success:
            last_fetch_observation = last_observation
        logger.info(
            f"[loop] {claim_id} iter {iter_num} done: "
            f"success={last_result.success} summary={last_result.summary[:200]}"
        )

        try:
            append_iteration(
                workspace, claim_id,
                iteration_num=last_observation.iter_num,
                action=getattr(last_observation.action, "value",
                                str(last_observation.action)),
                action_input=last_observation.input,
                observation_summary=last_observation.summary,
                success=last_observation.success,
            )
        except Exception as e:
            logger.warning(f"[loop] memory 기록 실패: {e}")

        # ★ Phase E: observation을 workspace에 저장 (runtime_agent가 Evidence 빌드용으로 read)
        try:
            obs_dict = {
                "iter_num": iter_num,
                "action": next_step.action.value,
                "input": next_step.input,
                "output": last_result.output,
                "summary": last_result.summary,
                "success": last_result.success,
                "error": last_result.error,
            }
            workspace.write_observation(
                claim_id,
                name=f"iter_{iter_num:02d}_{next_step.action.value}",
                data=obs_dict,
            )
        except Exception as e:
            logger.debug(f"[loop] observation 저장 실패: {e}")

        # ── FINISH 신호 감지 ──
        if last_result.output.get("_finish"):
            finished = True
            stop_reason = StopReason.COMPLETED
            logger.info(f"[loop] {claim_id}: FINISH 신호 감지, iter {iter_num}에서 종료")
            break

        # ── fail_fast 모드 ──
        if loop_config.fail_fast and not last_result.success:
            logger.info(f"[loop] {claim_id}: fail_fast 모드, iter {iter_num}에서 실패 종료")
            stop_reason = StopReason.ERROR
            break

    # ── ★ Auto verdict synthesis: plan 소진 시 마지막 fetch observation 기반 ──
    if not finished and plan_exhausted:
        auto_verdict = _synthesize_verdict_from_observation(
            plan=plan,
            claim=claim,
            claim_id=claim_id,
            last_observation=last_observation,
            iter_num=iter_num,
            tolerance=loop_config.value_match_tolerance,
        )
        if auto_verdict is not None:
            try:
                workspace.write_verdict(claim_id, auto_verdict.model_dump(mode="json"))
            except Exception as e:
                logger.debug(f"[loop] auto verdict 저장 실패: {e}")
            # [v6.21] 검증된 수치를 job 공유 저장소에 기록 — 다음 claim이 재사용.
            _save_verified_facts(workspace, auto_verdict, claim_id)
            v_str = getattr(auto_verdict.verdict, "value", str(auto_verdict.verdict))
            logger.info(
                f"[loop] {claim_id}: auto-synthesized verdict={v_str} "
                f"confidence={auto_verdict.confidence:.2f} (Phase D deterministic)"
            )
            return auto_verdict

    # ── 종료 처리 (auto synthesis 실패 또는 다른 종료) ──
    if not finished:
        # FINISH 호출 안 됨, auto synthesis도 실패 → 강제 unverifiable
        verdict = AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2,
            explanation=(
                f"Agent loop이 verdict 결정 없이 종료됨 (stop_reason={stop_reason.value}, "
                f"iter={iter_num}/{loop_config.max_iterations}). "
                f"마지막 observation: {last_observation.summary if last_observation else '(없음)'}"
            ),
            data_points=[],  # agent_loop 본문 — fetch evidence 미정의 구간
            iterations_used=iter_num,
            stop_reason=stop_reason,
        )
        try:
            workspace.write_verdict(claim_id, verdict.model_dump(mode="json"))
        except Exception as e:
            logger.debug(f"[loop] 강제 verdict 저장 실패: {e}")
        logger.info(
            f"[loop] {claim_id}: 강제 unverifiable 종료. stop_reason={stop_reason.value}"
        )
        return verdict

    # ── 정상 종료: workspace에서 verdict 읽기 (FinishTool이 저장한 것) ──
    try:
        verdict_data = workspace.read_verdict(claim_id)
        verdict = AgentVerdict(**verdict_data)
    except Exception as e:
        logger.warning(f"[loop] verdict 읽기 실패: {e}, 마지막 result에서 복원")
        verdict = AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            explanation="verdict.json 읽기 실패. 마지막 result에서 복원.",
            data_points=[],  # agent_loop 본문 — fetch evidence 미정의 구간
            iterations_used=iter_num,
            stop_reason=stop_reason,
        )

    # ── B2 sanity check (1-2 패치): LLM의 MATCH를 합성 verdict로 검증 ──
    # LLM이 fetch한 값을 무시하고 article 값을 그대로 답으로 박는 hallucination
    # 차단. fetch_evidence success가 있었으면 거기서 본 value vs claim value를
    # 객관적으로 비교(_synthesize_verdict_from_observation)해서, LLM의 MATCH가
    # 합성 결과와 어긋나면 합성 verdict로 덮어쓴다. 다른 verdict 종류(MISMATCH /
    # PARTIAL / UNVERIFIABLE)는 false-positive 위험이 낮아 일단 LLM 의견을 존중.
    if (
        verdict.verdict == VerdictType.MATCH
        and last_fetch_observation is not None
    ):
        synth = _synthesize_verdict_from_observation(
            plan=plan,
            claim=claim,
            claim_id=claim_id,
            last_observation=last_fetch_observation,
            iter_num=iter_num,
            tolerance=loop_config.value_match_tolerance,
        )
        if synth is not None and synth.verdict != VerdictType.MATCH:
            logger.warning(
                f"[loop] {claim_id}: LLM finish verdict=match vs 합성 verdict="
                f"{synth.verdict.value} 불일치 → 합성으로 정정. "
                f"(LLM explanation: {(verdict.explanation or '')[:120]!r})"
            )
            corrected = AgentVerdict(
                claim_id=verdict.claim_id,
                verdict=synth.verdict,
                confidence=synth.confidence,
                explanation=(
                    f"[자동 정정] LLM은 '일치'로 보고했으나, 조회된 KOSIS 값으로 "
                    f"객관 비교 시 결론이 다릅니다.\n\n"
                    f"합성 판정 근거: {synth.explanation}\n\n"
                    f"원래 LLM 설명(참고): {(verdict.explanation or '')[:200]}"
                ),
                data_points=synth.data_points or verdict.data_points,
                iterations_used=iter_num,
                stop_reason=verdict.stop_reason,
            )
            try:
                workspace.write_verdict(claim_id, corrected.model_dump(mode="json"))
            except Exception as e:
                logger.debug(f"[loop] 정정 verdict 저장 실패: {e}")
            verdict = corrected

    v_str = getattr(verdict.verdict, "value", str(verdict.verdict))
    logger.info(
        f"[loop] {claim_id}: 완료. verdict={v_str} "
        f"confidence={verdict.confidence:.2f} iterations={iter_num}"
    )
    return verdict