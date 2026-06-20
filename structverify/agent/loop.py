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
# [리팩] Step 8 판정 → verification.decide_verdict(agent)
from structverify.verification.adapters import (
    VerdictDecision,
    from_agent_calculate,
    from_agent_fetch,
)
from structverify.verification.decide_verdict import decide_verdict
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

    value_match_tolerance: float = 0.05
    """[2026-05-21 완화] auto verdict 합성 시 값 매칭 허용 오차 (5% 기본).
    기존 1%는 너무 strict해서 schema=0.79 vs fetch=0.8 (1.3% 차이) 같은
    실질적 일치도 mismatch로 떨어졌음. KOSIS 데이터의 통계적 변동/시점 차이를
    감안해 5%로 완화. 더 strict한 매칭이 필요하면 호출자가 인자로 override."""


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

# ── Auto verdict synthesis → verification.decide_verdict(agent) ─────────

def _evidence_to_data_points(evidence: dict, claim: Any) -> list[DataPointSpec]:
    """fetch observation evidence dict → DataPointSpec 리스트.

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
    population = getattr(schema, "population", None) if schema else None
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
    workspace: Any, verdict: Any, claim_id: str, claim: Any | None = None,
) -> None:
    """[v6.21] verdict의 data_points에서 검증된 수치를 job 공유 저장소에 기록.

    MATCH/MISMATCH verdict는 KOSIS 공식 수치를 data_points에 담는다.
    그 (indicator, time_period, value, unit)을 verified_facts에 저장하면,
    다음 claim이 같은 수치를 catalog_search 없이 재사용할 수 있다.

    UNVERIFIABLE은 공식 수치가 없으므로 저장하지 않는다.

    [S 패치 2026-05-21] claim이 전달되면 sent_id 기반 sibling_evidence에도 같이
    기록해 같은 sent_id의 형제 sub-claim들이 활용할 수 있도록 한다.
    """
    try:
        v_type = getattr(verdict.verdict, "value", str(verdict.verdict))
        if v_type not in ("match", "mismatch"):
            return  # 검증 실패 — 신뢰할 수치 없음
        dps = getattr(verdict, "data_points", None) or []

        # sibling_evidence용 sent_id / value_role 추출
        sent_id = ""
        role = ""
        if claim is not None:
            sent_id = str(getattr(claim, "sent_id", "") or "").strip()
            schema = getattr(claim, "schema", None)
            role = (getattr(schema, "value_role", None) or "") if schema else ""

        for dp in dps:
            val = getattr(dp, "resolved_value", None)
            if val is None:
                continue
            fact = {
                "indicator": getattr(dp, "indicator", "") or "",
                "time_period": (
                    getattr(dp, "source_time", None) or getattr(dp, "time", "") or ""
                ),
                # [2026-05-21] population 추가 — sub-claim별 격리 위해 캐시 키에 포함
                "population": (getattr(dp, "population", None) or ""),
                "value": val,
                "unit": getattr(dp, "resolved_unit", None) or "",
                "source": getattr(dp, "source", None) or "KOSIS",
                "claim_id": str(claim_id),
                "verdict": v_type,
            }
            workspace.append_verified_fact(fact)
            if sent_id and role:
                workspace.record_sibling_evidence(
                    sent_id=sent_id, role=role, evidence=fact,
                )
    except Exception as e:
        logger.debug(f"[loop] verified_fact 저장 실패 (무시): {e}")


def _verdict_decision_to_agent_verdict(
    decision: VerdictDecision,
    evidence: dict,
    claim: Any,
    iter_num: int,
    stop_reason: StopReason = StopReason.COMPLETED,
) -> AgentVerdict:
    """VerdictDecision → AgentVerdict (data_points·iter 메타 포장)."""
    return AgentVerdict(
        claim_id=decision.claim_id,
        verdict=decision.verdict,
        confidence=decision.confidence,
        explanation=decision.explanation,
        data_points=_evidence_to_data_points(evidence, claim),
        iterations_used=iter_num,
        stop_reason=stop_reason,
    )


def _synthesize_verdict_from_observation(
    plan: Plan,
    claim: Any,
    claim_id: str,
    last_observation: Observation | None,
    iter_num: int,
    tolerance: float,
    all_fetch_observations: list | None = None,
    config: dict | None = None,
) -> AgentVerdict | None:
    """Plan steps 소진 시 fetch observation 보고 deterministic verdict 합성.

    Phase D의 임시 verdict 결정 로직. Phase E에서 LLM 기반 verdict로 교체 예정.
    [리팩] 판정 본문은 verification.decide_verdict(profile=agent)에 위임.
    """
    if last_observation is None:
        return None
    if last_observation.action != ActionType.FETCH_EVIDENCE:
        return None

    normalized, early = from_agent_fetch(
        claim,
        last_observation,
        plan,
        tolerance=tolerance,
        all_fetch_observations=all_fetch_observations,
    )
    if early is not None:
        return _verdict_decision_to_agent_verdict(early, {}, claim, iter_num)
    if normalized is None:
        return None

    decision = decide_verdict(claim, normalized, config or {}, profile="agent")
    return _verdict_decision_to_agent_verdict(
        decision, normalized.evidence, claim, iter_num,
    )


def _synthesize_verdict_from_calculate(
    plan: Plan,
    claim: Any,
    claim_id: str,
    last_calc_observation: Observation | None,
    iter_num: int,
    last_fetch_observation: Observation | None = None,
    workspace: Any = None,
    config: dict | None = None,
) -> AgentVerdict | None:
    """[패치] Plan 소진 + 마지막 성공한 관측이 CALCULATE인 경우 verdict 합성.

    LLM이 prev/current를 계산했지만 finish를 안 부르고 다시 같은 액션 반복
    → 중복차단 → 강제 unverifiable로 죽는 케이스 회복.
    [리팩] 판정 본문은 verification.decide_verdict(profile=agent)에 위임.
    """
    normalized, _early = from_agent_calculate(
        claim,
        last_calc_observation,
        plan,
        last_fetch_observation=last_fetch_observation,
        workspace=workspace,
    )
    if normalized is None:
        return None

    decision = decide_verdict(claim, normalized, config or {}, profile="agent")
    return _verdict_decision_to_agent_verdict(decision, {}, claim, iter_num)


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

    # [2026-05-21] 사전 가드 — claim에 검증 가능한 value가 *없으면* 즉시 unverifiable.
    # LLM(schema_inductor)이 한 문장에서 정상 schema + 빈(value=null) schema를 함께
    # 만들어 별도 sub-claim으로 분기되던 케이스 회귀 방지. 빈 schema는 fetch를
    # 아무리 해도 비교 불가 → max_iter까지 reflect 헛돌이만 발생.
    # aggregation은 별도 흐름(N개 시점 fetch → calc)이라 value 없어도 OK.
    _claim_schema = getattr(claim, "schema", None)
    _claim_role = getattr(_claim_schema, "value_role", None) if _claim_schema else None
    _claim_value = getattr(_claim_schema, "value", None) if _claim_schema else None
    if (
        _claim_schema is not None
        and _claim_value is None
        and _claim_role != "aggregation"
    ):
        logger.warning(
            f"[loop] {claim_id}: schema.value=None (role={_claim_role!r}) — "
            f"검증 대상 수치가 없어 즉시 unverifiable. "
            f"indicator={getattr(_claim_schema, 'indicator', None)!r}, "
            f"time={getattr(_claim_schema, 'time_period', None)!r}, "
            f"population={getattr(_claim_schema, 'population', None)!r}"
        )
        return AgentVerdict(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2,
            explanation=(
                "이 sub-claim의 schema에 비교할 수치(value)가 없어 검증 불가. "
                "원문에서 수치 추출이 실패했거나, 같은 문장의 다른 sub-claim에서 "
                "수치가 모두 표현된 경우."
            ),
            data_points=[],
            iterations_used=0,
            stop_reason=StopReason.COMPLETED,
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
    # [패치 H-3] 같은 claim의 모든 성공 fetch observation을 모음.
    # 마지막 fetch가 prev_time만 받았을 때, 이전 fetch의 rows[]에서 claim_time
    # 시점 row를 찾아 비교/계산할 수 있도록 한다. 또 prev/current row가 서로
    # 다른 fetch에서 와도 같은 지표(matched_row criteria)로 묶어 짝지을 수 있음.
    all_fetch_observations: list[Observation] = []
    # [패치] LLM이 계산까지 했는데 finish를 안 부르고 중복차단으로 죽는
    # 케이스 대응 — 마지막으로 성공한 calculate observation을 별도 추적.
    last_calc_observation: Observation | None = None
    last_result: ToolResult | None = None
    finished = False
    stop_reason = StopReason.MAX_ITERATIONS
    plan_step_idx = 0
    plan_exhausted = False

    # ── [중복 action 대응] reflect 모드 전용 ─────────────────────────
    # reflect(HCX)가 thought엔 "다른 검색어"라 쓰면서 action.input.query는
    # 동일하게 두는 일이 잦다. 이전엔 *강제 종료*했으나, 이는 fetch fail
    # 후 retry 기회를 박탈해 진짜 정답에 도달 못 하는 부작용이 컸음
    # (project_fetch_lockup, 2026-05-26).
    #
    # 새 정책: 중복 감지 시 *이전 observation을 그대로 반환 + 다음 행동 hint*.
    # 연속 N회 이상 지속되면 강제 다음 action 또는 종료.
    #   1회 중복: cached observation + soft hint
    #   2회 중복: cached observation + strong hint (다음 action 명시)
    #   3회 이상: 헛돌이 — 종료
    _seen_action_keys: set[str] = set()
    _action_key_to_observation: dict[str, Observation] = {}
    _consecutive_dup = 0
    _DUP_HARD_LIMIT = 3  # 연속 N회 도달 시 종료

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
                # [S 패치 2026-05-21] 같은 sent_id의 sibling base evidence를
                # memory_text 상단에 inject → derived claim이 추가 fetch 없이
                # base의 KOSIS 값을 활용해 즉시 calculate 가능.
                try:
                    _schema = getattr(claim, "schema", None)
                    _role = (getattr(_schema, "value_role", None) or "") if _schema else ""
                    _sent_id = str(getattr(claim, "sent_id", "") or "").strip()
                    if _role in ("derived_rate", "derived_difference") and _sent_id:
                        _sibs = workspace.read_sibling_evidence(_sent_id) or []
                        _base_sibs = [s for s in _sibs if s.get("role") == "base"]
                        if _base_sibs and iter_num == 1:
                            # 첫 iter에만 inject (이후엔 last_observation으로 전달됨)
                            _sib_lines = []
                            for _s in _base_sibs:
                                _sib_lines.append(
                                    f"  - role={_s.get('role')!r} "
                                    f"indicator={_s.get('indicator')!r} "
                                    f"value={_s.get('value')} "
                                    f"unit={_s.get('unit')!r} "
                                    f"time_period={_s.get('time_period')!r} "
                                    f"source={_s.get('source')!r}"
                                )
                            _sib_block = (
                                "## 같은 sent_id의 sibling base 검증 결과 (S 패치)\n"
                                "이 derived claim과 같은 문장에서 분기된 *base sub-claim*이\n"
                                "KOSIS에서 이미 검증한 값:\n"
                                + "\n".join(_sib_lines) + "\n\n"
                                "★ 활용 방법:\n"
                                "  - 이 base value가 derived의 *current 시점* 값입니다.\n"
                                "  - claim.schema.prev_value가 원문에 있으면 그 값과 함께\n"
                                "    *추가 fetch 없이* calculate (또는 finish) 가능합니다.\n"
                                "  - prev fetch가 필요하면 같은 stat_id로 한 번만.\n\n"
                                "---\n\n"
                            )
                            memory_text = _sib_block + (memory_text or "")
                            logger.info(
                                f"[loop] {claim_id}: sibling base evidence "
                                f"{len(_base_sibs)}건 inject (sent_id={_sent_id!r})"
                            )
                except Exception as _e:
                    logger.debug(f"[loop] sibling inject 실패 (무시): {_e}")
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

        # ── [J 패치 2026-05-21] absolute claim에서 calculate 액션 차단 ──
        # plan.claim_type=ABSOLUTE인데 reflect LLM이 자율적으로 calculate를
        # 부르는 케이스 (출생아 수 base 20717명: fetch 후 자율 calc → unverifiable).
        # absolute는 단일 값 비교라 수식 계산이 필요 없음. tool 실행 스킵 +
        # "absolute니 finish 하라" observation 전달해 다음 iter에 finish 유도.
        # G 패치(prompt 가이드)로 LLM 자제 시도했으나 무시되는 케이스가 잦아
        # loop 단에서 결정론적 차단.
        if (
            loop_config.mode == "reflect"
            and plan.claim_type == ClaimType.ABSOLUTE
            and next_step.action == ActionType.CALCULATE
        ):
            logger.warning(
                f"[loop] {claim_id} iter {iter_num}: "
                f"plan.claim_type=ABSOLUTE인데 calculate 호출 — 스킵하고 finish 유도"
            )
            last_observation = Observation(
                iter_num=iter_num,
                action=next_step.action,
                input=next_step.input,
                output={},
                summary=(
                    "[absolute 가드] 이 claim은 plan_type=absolute (단일 값 검증)"
                    "이므로 calculate 호출이 불필요합니다. 직전 fetch_evidence "
                    "값이 claim.value와 일치하면 finish(match)로 종료하세요."
                ),
                success=False,
                error="absolute_calc_blocked",
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
                logger.warning(
                    f"[loop] absolute 가드 observation 기록 실패: {e}"
                )
            continue  # 다음 iter — reflect가 finish 결정하도록

        # ── [중복 action 대응] reflect 모드에서만 ──────────────────
        # 같은 (action, input)이 이미 실행됐으면 *cached observation 반환* +
        # 다음 행동 hint. 연속 N회면 종료.
        if loop_config.mode == "reflect":
            _akey = _action_key(next_step)
            if _akey in _seen_action_keys:
                _consecutive_dup += 1
                _cached_obs = _action_key_to_observation.get(_akey)
                logger.info(
                    f"[loop] {claim_id} iter {iter_num}: 중복 action 감지 "
                    f"(action={next_step.action.value}, 연속 {_consecutive_dup}회) "
                    f"— cached observation 재사용"
                )

                # cached observation summary 그대로 + hint 추가
                _base_summary = (
                    _cached_obs.summary if _cached_obs is not None
                    else f"이전 iter에서 같은 입력으로 실행된 적이 있습니다."
                )
                if _consecutive_dup == 1:
                    _hint = (
                        " | [hint] 이 입력은 직전에 이미 시도했고 동일한 결과가 나왔습니다. "
                        "결과를 다시 검토하고 *다음에 할 action을 결정*하세요. "
                        "(예: catalog_search 결과를 보고 fetch_evidence로 후보 시도, "
                        "또는 다른 query/category로 catalog_search 재호출)"
                    )
                elif _consecutive_dup == 2:
                    _hint = (
                        " | [strong hint] 동일 입력 2회 반복입니다. *반드시 다른 action* "
                        "또는 *다른 query/input*을 선택하세요. catalog_search 반복 금지 — "
                        "fetch_evidence(다른 candidate_id) 또는 explore_catalog를 쓰거나 "
                        "이미 충분한 정보가 있으면 finish(unverifiable)로 종료."
                    )
                else:
                    _hint = (
                        " | [final hint] 동일 입력 3회 이상 — 헛돌이로 판단합니다."
                    )

                last_observation = Observation(
                    iter_num=iter_num,
                    action=next_step.action,
                    input=next_step.input,
                    output=(_cached_obs.output if _cached_obs is not None else {}),
                    summary=_base_summary + _hint,
                    success=(_cached_obs.success if _cached_obs is not None else False),
                    error="duplicate_action_cached",
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
                if _consecutive_dup >= _DUP_HARD_LIMIT:
                    logger.warning(
                        f"[loop] {claim_id}: 중복 action {_consecutive_dup}회 연속 "
                        f"→ 헛돌이로 판단, iter {iter_num}에서 종료"
                    )
                    plan_exhausted = True
                    stop_reason = StopReason.MAX_ITERATIONS
                    break
                continue  # 다음 iter — reflect가 cached + hint 보고 결정
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
            current_plan=plan,  # [2026-05-26] replan tool이 원래 plan 참고용
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

        # ── [2026-05-26] replan tool 결과 처리 — plan 통째 교체 ──
        # ReplanTool이 성공하면 output에 new_plan(dict)이 들어있음.
        # 이걸 받아 *plan 객체 자체*를 새로 만들고 plan_step_idx, _seen_action_keys
        # 등 plan-관련 상태를 리셋. 이후 iter는 새 plan으로 진행.
        if (
            next_step.action == ActionType.REPLAN
            and last_result.success
            and isinstance(last_result.output, dict)
            and last_result.output.get("new_plan")
        ):
            try:
                _new_plan_dict = last_result.output["new_plan"]
                # 직렬화 안전성을 위해 claim_id는 기존 plan것 유지
                _new_plan_dict.setdefault("claim_id", str(plan.claim_id))
                _new_plan = Plan.model_validate(_new_plan_dict)
                _old_type = plan.claim_type.value
                _new_type = _new_plan.claim_type.value
                plan = _new_plan
                # plan-관련 상태 리셋
                plan_step_idx = 0
                plan_exhausted = False
                # 중복 차단 카운터 리셋 (새 plan이라 처음부터 다시)
                _seen_action_keys.clear()
                _consecutive_dup = 0
                logger.info(
                    f"[loop] {claim_id}: ★ plan 교체 완료 (replan) — "
                    f"claim_type: {_old_type} → {_new_type}, "
                    f"new steps: {len(plan.initial_steps)}"
                )
            except Exception as _e:
                logger.warning(f"[loop] {claim_id}: replan 결과 plan 파싱 실패: {_e}")

        # ── Observation 생성 + memory 기록 ──
        # [P28 2026-05-22] T2 — fetch_evidence 실패(no row matched/not relevant 등) 시
        # observation summary에 fallback hint 주입 → 다음 reflect turn이 catalog_search를
        # 2단계 fallback으로 재호출하도록 유도.
        # [P30 2026-05-22] 2단계 fallback hint:
        #   1. query_rewrite=true — LLM이 표 이름 친화 어휘로 query 변형 후 재검색.
        #      catalog 후보에 정답 표가 진입조차 못 한 경우 회복.
        #   2. force_explore=true (explore_mode=meta 기본) — top 후보 표의 항목/분류
        #      메타(ITM/OBJ)를 받아 LLM이 정답 표 식별. 정답이 후보에 있으나 cosine
        #      점수가 낮아 묻힌 경우 회복.
        _summary = last_result.summary
        if (
            next_step.action == ActionType.FETCH_EVIDENCE
            and not last_result.success
            and loop_config.mode == "reflect"
        ):
            _summary = (
                _summary
                + " | [hint] 이 표에서 row를 찾지 못했습니다. 다음 turn에 catalog_search "
                "재호출 시 *두 옵션 중 하나*를 추가하세요:"
                "\n  (a) input에 \"query_rewrite\": true — LLM이 검색어를 표 이름 친화 "
                "어휘로 변형(예: '체외 충격파 쇄석술 장비' → '시군구별 의료장비'). "
                "원본 query가 row 항목 키워드일 때 정답 표가 후보에 진입하도록 도움."
                "\n  (b) input에 \"force_explore\": true — top 표들의 *항목 메타*(ITM/OBJ)를 "
                "가져와 LLM이 '이 표에 indicator가 직접 있는가' 판단. 정답이 후보에는 "
                "있으나 점수가 낮아 묻힌 경우 회복."
                "\n  먼저 (a)를 시도하고 그래도 부족하면 (b). 둘 다 한 번에 켜도 OK."
                "\n  ★ (a)/(b)도 *전부* 시도했는데 또 fail이고, 받은 표 sample을 봐도 "
                "claim의 정확한 값이 row로 없으면 → *replan* action 호출. "
                "(예: claim '증가 수 52'인데 표엔 절대값만 — 계산 필요)"
            )

        last_observation = Observation(
            iter_num=iter_num,
            action=next_step.action,
            input=next_step.input,
            output=last_result.output,
            summary=_summary,
            success=last_result.success,
            error=last_result.error,
        )
        # [중복 action 대응] 같은 (action, input) 재호출 시 cache 재사용
        if loop_config.mode == "reflect":
            try:
                _action_key_to_observation[_action_key(next_step)] = last_observation
            except Exception:
                pass
        # B2 sanity check용 — fetch가 성공할 때마다 갱신 (finish가 덮어쓰지 못하게)
        if next_step.action == ActionType.FETCH_EVIDENCE and last_result.success:
            last_fetch_observation = last_observation
            # [패치 H-3] aggregate pool에도 추가
            all_fetch_observations.append(last_observation)
        # [패치] calculate가 성공할 때마다 갱신 — finish 미호출 시 합성 verdict용
        if next_step.action == ActionType.CALCULATE and last_result.success:
            last_calc_observation = last_observation
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
        # [2026-05-27] reflect LLM의 thought(자연어 사고)·confidence·proposed_verdict를
        # observation에 함께 보존 → 프론트 실시간 카드/콘솔이 "왜 이 action을 골랐는지"를
        # 표시할 수 있게. decision은 reflect 모드에서 LLM이 반환한 ReflectDecision이고,
        # deterministic 모드(plan 그대로)에선 None — 그 경우 next_step.rationale(=planner가
        # 적은 rationale)이라도 노출.
        try:
            _thought = None
            _conf_so_far = None
            _proposed_verdict = None
            if 'decision' in locals() and decision is not None:
                _thought = getattr(decision, "thought", None) or None
                _conf_raw = getattr(decision, "confidence_so_far", None)
                if isinstance(_conf_raw, (int, float)):
                    _conf_so_far = float(_conf_raw)
                _pv = getattr(decision, "proposed_verdict", None)
                if _pv is not None:
                    _proposed_verdict = getattr(_pv, "value", str(_pv))
            # reflect가 thought를 안 줬으면 planner가 짠 step.rationale을 fallback으로
            _thought = _thought or (next_step.rationale or None)
            obs_dict = {
                "iter_num": iter_num,
                "action": next_step.action.value,
                "input": next_step.input,
                "output": last_result.output,
                "summary": last_result.summary,
                "success": last_result.success,
                "error": last_result.error,
                # ↓ LLM 자연어 사고 (프론트가 "LLM 생각" 라인으로 노출)
                "thought": _thought,
                "confidence_so_far": _conf_so_far,
                "proposed_verdict": _proposed_verdict,
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

        # ── [패치 2026-05-21] calculate 성공 후 자동 finish 트리거 ──
        # growth_rate/difference 같은 derived claim에서 prev/current를 fetch로
        # 다 모은 다음 calculate까지 성공시켰는데 reflect LLM이 finish는 안
        # 부르고 다음 iter에 *또* fetch_evidence를 시도하는 헛돌이가 잦다
        # (2026-05-21 진단: 출생아 수 증가율 claim이 iter 3에서 9.19% 계산
        # 끝났는데 iter 4·5·6에서 또 fetch → 다른 시점 row 끌어와 결과 변동).
        # _synthesize_verdict_from_calculate의 가드(derived suffix + fetch
        # 1건 이상)를 통과해 match/mismatch가 명확하면 그 자리에서 verdict
        # 확정. UNVERIFIABLE이면 통과시키지 않아 LLM이 더 fetch할 기회 유지.
        # [2026-05-21] sibling base cache가 current를 공급하는 경우 fetch는 prev 1건만
        # 들어와도 calculate에 필요한 두 값이 다 모인 것. >=2 가드를 그대로 두면
        # 2번째 fetch가 중복으로 차단된 케이스(연속 dup)에서 auto-finish가 미발화 →
        # LLM이 엉뚱한 표(예: 혼인건수)로 추가 fetch를 시도해 최종 verdict 오염.
        # sent_id의 sibling base evidence가 있으면 fetch 1건도 충분으로 간주.
        _has_sibling_base = False
        try:
            _schema = getattr(claim, "schema", None)
            _role = (getattr(_schema, "value_role", None) or "") if _schema else ""
            _sent_id = str(getattr(claim, "sent_id", "") or "").strip()
            if (
                _role in ("derived_rate", "derived_difference")
                and _sent_id
                and hasattr(workspace, "read_sibling_evidence")
            ):
                _sibs = workspace.read_sibling_evidence(_sent_id) or []
                _has_sibling_base = any(s.get("role") == "base" for s in _sibs)
        except Exception:
            _has_sibling_base = False
        _fetch_threshold = 1 if _has_sibling_base else 2

        # [P22 2026-05-22] sibling base가 있으면 fetch 0번도 OK.
        # LLM이 verified_facts/sibling 캐시만으로 calculate(current=sibling, prev=캐시)를
        # 호출하는 케이스 — fetch_observation이 None이라 기존 gate가 막아 unverifiable로
        # 떨어지던 회귀(혼인 건수 증가율 4.9% claim 케이스). sibling이 있으면 last_fetch
        # 없어도 calculate 결과를 신뢰.
        _gate_pass = (
            not finished
            and last_result.success
            and next_step.action == ActionType.CALCULATE
            and last_calc_observation is not None
        )
        if _has_sibling_base:
            # sibling 있으면 fetch threshold/observation 요구 없음
            _gate_pass = _gate_pass
        else:
            _gate_pass = _gate_pass and (
                last_fetch_observation is not None
                and len(all_fetch_observations) >= _fetch_threshold
            )
        if _gate_pass:
            early_verdict = _synthesize_verdict_from_calculate(
                plan=plan,
                claim=claim,
                claim_id=claim_id,
                last_calc_observation=last_calc_observation,
                iter_num=iter_num,
                last_fetch_observation=last_fetch_observation,
                workspace=workspace,
                config=config,
            )
            if (
                early_verdict is not None
                and early_verdict.verdict != VerdictType.UNVERIFIABLE
            ):
                try:
                    workspace.write_verdict(
                        claim_id, early_verdict.model_dump(mode="json"),
                    )
                except Exception as e:
                    logger.debug(f"[loop] early calculate verdict 저장 실패: {e}")
                _save_verified_facts(workspace, early_verdict, claim_id, claim=claim)
                v_str = getattr(
                    early_verdict.verdict, "value", str(early_verdict.verdict),
                )
                logger.info(
                    f"[loop] {claim_id}: iter {iter_num} calculate 성공 후 "
                    f"finish 자동 트리거 (verdict={v_str} "
                    f"conf={early_verdict.confidence:.2f}) — "
                    f"reflect의 finish 미호출 헛돌이 차단"
                )
                return early_verdict

        # ── fail_fast 모드 ──
        if loop_config.fail_fast and not last_result.success:
            logger.info(f"[loop] {claim_id}: fail_fast 모드, iter {iter_num}에서 실패 종료")
            stop_reason = StopReason.ERROR
            break

    # ── ★ Auto verdict synthesis ──
    # [패치 K] plan_exhausted 뿐 아니라 max_iter 자연 종료 시에도 합성 시도.
    # reflect 모드는 plan_step_idx를 안 쓰므로 plan_exhausted=False인 채로
    # max_iterations에 도달하면, fetch는 성공했는데 LLM이 finish를 안 부르고
    # 헛돌이로 iter을 다 써버린 경우 synthesis 자체가 안 발화해 unverifiable
    # 기본값으로 떨어지던 버그 (혼인 건수 base 케이스). last_fetch_observation
    # 이 살아있으면 그걸로 합성 시도.
    if not finished and (plan_exhausted or last_fetch_observation is not None):
        # [패치 G] last_observation 대신 last_fetch_observation(마지막 *성공한*
        # fetch)을 사용한다. 중복 차단으로 plan_exhausted가 끝나는 경우
        # last_observation은 success=False 더미이고, 그걸 그대로 합성에 넣으면
        # _synthesize_verdict_from_observation의 "fetch 실패" 분기로 떨어져
        # UNVERIFIABLE conf=0.25가 박힌다. 실제로는 직전 iter에 성공한 fetch
        # 값(20787, 0.8, 18919 등)이 claim과 거의 일치하는데도 그 비교가
        # 발화하지 않아 unverifiable로 끝나는 버그.
        synth_target = (
            last_fetch_observation if last_fetch_observation is not None
            else last_observation
        )
        auto_verdict = _synthesize_verdict_from_observation(
            plan=plan,
            claim=claim,
            claim_id=claim_id,
            last_observation=synth_target,
            iter_num=iter_num,
            tolerance=loop_config.value_match_tolerance,
            all_fetch_observations=all_fetch_observations,
            config=config,
        )
        # [패치] fetch 기반 합성이 None이거나 UNVERIFIABLE인데 LLM이 계산을
        # 끝낸 케이스면 calculate 결과로 다시 합성 시도. calculate 결과가
        # 더 명확한(match/mismatch) verdict면 그쪽을 우선한다. fetch 합성이
        # "prev row 없어 직접 계산 불가 → unverifiable"로 떨어질 때, agent가
        # 따로 calculate로 9.193% 같은 결과를 내놨으면 그걸 살려야 함.
        if last_calc_observation is not None and (
            auto_verdict is None
            or auto_verdict.verdict == VerdictType.UNVERIFIABLE
        ):
            calc_verdict = _synthesize_verdict_from_calculate(
                plan=plan,
                claim=claim,
                claim_id=claim_id,
                last_calc_observation=last_calc_observation,
                iter_num=iter_num,
                last_fetch_observation=last_fetch_observation,
                workspace=workspace,
                config=config,
            )
            if calc_verdict is not None and (
                calc_verdict.verdict != VerdictType.UNVERIFIABLE
                or auto_verdict is None
            ):
                auto_verdict = calc_verdict
        if auto_verdict is not None:
            try:
                workspace.write_verdict(claim_id, auto_verdict.model_dump(mode="json"))
            except Exception as e:
                logger.debug(f"[loop] auto verdict 저장 실패: {e}")
            # [v6.21] 검증된 수치를 job 공유 저장소에 기록 — 다음 claim이 재사용.
            _save_verified_facts(workspace, auto_verdict, claim_id, claim=claim)
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
    # 합성 결과 MISMATCH이면 합성으로 덮어쓴다.
    # [패치 2026-05-20] 합성이 UNVERIFIABLE인 경우엔 정정하지 않는다. fetch가
    # 단일 시점만 받아 합성이 prev/cur 비교 불가로 UNVERIFIABLE 떨어지는
    # 케이스에서, LLM이 (KOSIS prev + article current) 같이 섞어 계산한 8.82%
    # vs article 8.7%처럼 합리적인 결론을 잘못 강등시키는 걸 방지. evidence
    # 0건 자체는 A안 가드(FinishTool)에서 이미 차단됨.
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
            all_fetch_observations=all_fetch_observations,
            config=config,
        )
        if synth is not None and synth.verdict == VerdictType.MISMATCH:
            logger.warning(
                f"[loop] {claim_id}: LLM finish verdict=match vs 합성 verdict="
                f"{synth.verdict.value} (MISMATCH) → 합성으로 정정. "
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

    # ── [N 패치 2026-05-21] LLM의 MISMATCH / UNVERIFIABLE도 합성 sanity check ──
    # LLM이 sub-claim 단위 검증해야 하는데 (1) claim_text 전체의 비교 문맥까지
    # 따져 mismatch 박거나 (2) fetch evidence 충분히 있는데도 unverifiable 박는
    # 케이스가 잦다.
    # (1) 경기 11573 vs fetch 11573 완벽 매치인데 mismatch 박힘
    # (2) 혼인 base 18921 vs fetch 18919 (0.01% 차이)인데 unverifiable 박힘
    # 합성 verdict가 MATCH이면 LLM 결정이 잘못 — 합성으로 정정.
    if (
        verdict.verdict in (VerdictType.MISMATCH, VerdictType.UNVERIFIABLE)
        and last_fetch_observation is not None
    ):
        synth = _synthesize_verdict_from_observation(
            plan=plan,
            claim=claim,
            claim_id=claim_id,
            last_observation=last_fetch_observation,
            iter_num=iter_num,
            tolerance=loop_config.value_match_tolerance,
            all_fetch_observations=all_fetch_observations,
            config=config,
        )
        if synth is not None and synth.verdict == VerdictType.MATCH:
            _orig_v = verdict.verdict.value
            logger.warning(
                f"[loop] {claim_id}: LLM finish verdict={_orig_v} vs 합성 "
                f"verdict=MATCH → 합성으로 정정 (sub-claim의 schema.value와 "
                f"KOSIS 조회값이 객관 일치). "
                f"(LLM explanation: {(verdict.explanation or '')[:120]!r})"
            )
            corrected = AgentVerdict(
                claim_id=verdict.claim_id,
                verdict=synth.verdict,
                confidence=synth.confidence,
                explanation=(
                    f"[자동 정정] LLM은 '{_orig_v}'(으)로 보고했으나, 조회된 "
                    f"KOSIS 값이 claim.value와 객관 일치합니다.\n\n"
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

    # [S 패치] FinishTool 정상 경로의 verdict도 sibling_evidence에 기록 →
    # 같은 sent_id의 derived sub-claim들이 활용할 수 있도록.
    # auto-synthesis 경로는 위쪽 _save_verified_facts에서 이미 기록되므로
    # 여기선 *finished + match/mismatch* 케이스만 추가 처리.
    if finished:
        _save_verified_facts(workspace, verdict, claim_id, claim=claim)

    v_str = getattr(verdict.verdict, "ovalue", str(verdict.verdict))
    logger.info(
        f"[loop] {claim_id}: 완료. verdict={v_str} "
        f"confidence={verdict.confidence:.2f} iterations={iter_num}"
    )
    return verdict