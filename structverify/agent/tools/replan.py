"""
structverify.agent.tools.replan — Plan 자체를 재작성하는 tool.

배경:
  기존 fallback 메커니즘(try_ids, catalog retry, row_matcher rescue 등)은 모두
  *같은 plan 안에서* 데이터를 찾는 시도. claim 값이 표에 *직접 row로 없는*
  경우(예: "강원도 의료장비 증가 수 52"는 절대값 row가 아니라 current-prev delta)
  에는 어떤 fallback도 못 잡음.

  Replan tool은 이런 *구조적 mismatch*를 잡음. fetch 모두 실패한 상황에서:
    1. 표 sample/메타를 다시 확인
    2. claim 값이 표의 row로 존재하는지 vs 계산 대상인지 LLM이 판단
    3. 필요 시 *plan 자체*를 새로 생성 (claim_type 변경 + 새 steps)

  loop이 새 plan을 받아 *이후 iteration*에서 사용.

호출 조건:
  - 모든 try_ids 실패 + 모든 catalog retry 소진 후
  - reflect prompt에서 명시적으로 'replan' 호출 가이드
  - per-claim 최대 2회 (무한 replan 방지 — workspace에서 카운트)

기존 fallback과의 관계:
  ★ 기존 fallback은 *plan 안의 답 찾기* — 그대로 작동
  ★ replan은 *plan을 갈아끼우기* — 위에 얹는 한 층
  → 서로 잡아먹지 않음. replan 호출 후 새 plan으로 다시 fallback 시도 가능.
"""
from __future__ import annotations

from typing import Any

from structverify.agent.schemas import ActionType
from structverify.utils.logger import get_logger

from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


# per-claim replan 호출 카운트 키 (workspace observation 파일 이름)
_REPLAN_COUNT_OBS_KEY = "_replan_count"
_REPLAN_MAX_PER_CLAIM = 2


def _read_replan_count(workspace, claim_id) -> int:
    """현재 claim의 replan 호출 누적 횟수."""
    if workspace is None:
        return 0
    try:
        data = workspace.read_observation(claim_id, _REPLAN_COUNT_OBS_KEY)
        if isinstance(data, dict):
            return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def _write_replan_count(workspace, claim_id, count: int) -> None:
    if workspace is None:
        return
    try:
        workspace.write_observation(claim_id, _REPLAN_COUNT_OBS_KEY, {"count": count})
    except Exception as e:
        logger.debug(f"[replan] count 저장 실패: {e}")


def _collect_observation_summary(workspace, claim_id) -> list[dict]:
    """현재까지의 observation들에서 fetch 결과 + catalog 결과 요약 추출.

    planner LLM에게 '여태 무엇을 시도했고 어떤 데이터가 있었는지' 보여주는 용도.
    """
    if workspace is None:
        return []
    try:
        names = sorted(workspace.list_observations(claim_id))
    except Exception:
        return []

    summaries: list[dict] = []
    for name in names:
        # 내부 메타 파일 skip
        if name.startswith("_"):
            continue
        try:
            data = workspace.read_observation(claim_id, name)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        # observation은 보통 {iter_num, action, input, output, summary, success, error}
        action = data.get("action")
        summary = data.get("summary", "")
        success = data.get("success")
        output = data.get("output") or {}
        entry: dict[str, Any] = {
            "obs": name,
            "action": action,
            "success": success,
            "summary": str(summary)[:300],
        }
        # fetch_evidence 성공/실패 시 표 sample도 추출
        if action == ActionType.FETCH_EVIDENCE.value:
            ev = output.get("evidence")
            if isinstance(ev, dict):
                entry["fetched_value"] = ev.get("value")
                entry["fetched_unit"] = ev.get("unit")
                entry["fetched_time"] = ev.get("time_period")
                entry["stat_id"] = output.get("candidate_id")
            elif ev is None:
                entry["fetched_value"] = None
                entry["stat_id"] = output.get("candidate_id")
                entry["tried_candidates"] = output.get("tried_candidates")
        # catalog_search candidates 요약
        elif action == ActionType.CATALOG_SEARCH.value:
            cands = output.get("candidates") or []
            entry["candidates_top3"] = [
                {"id": c.get("id"), "name": c.get("name"), "score": c.get("score")}
                for c in cands[:3] if isinstance(c, dict)
            ]
        summaries.append(entry)
    return summaries


@register_tool(ActionType.REPLAN)
class ReplanTool(ToolBase):
    """Plan 재작성 tool.

    호출 트리거: 모든 catalog 후보 fetch 실패 + retry 소진 후.
    동작: 지금까지의 observation을 planner LLM에 보여주고 *새 plan* 생성 요청.
    효과: loop이 새 plan으로 갈아끼우고 *이후* iter에서 새 claim_type/steps 사용.
    """

    name = ActionType.REPLAN
    description = (
        "Plan 자체를 새로 만든다. claim 값이 표에 *직접 row로 없고* 계산이 "
        "필요한 경우(예: '증가 수', '증감률' 등 delta/derived 지표인데 plan이 "
        "absolute로 잡혀있는 경우) 사용. **호출 조건**: 모든 fetch 후보 실패 + "
        "catalog 재검색도 소진된 후에만. 일반 retry는 catalog_search나 fetch_evidence "
        "재호출로 처리하고, replan은 *마지막 수단*."
    )
    input_schema = {
        "reason": (
            "왜 replan이 필요한지 한 줄 설명. 예: '모든 후보 표에 절대값만 있고 "
            "claim의 변화량 값(52)이 row에 없음 → 차이 계산으로 변경 필요'"
        ),
    }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        reason = str(input_data.get("reason") or "").strip()
        workspace = getattr(context, "workspace", None)
        claim_id = getattr(context, "claim_id", None)
        claim = getattr(context, "claim", None)

        if claim is None:
            return ToolResult(
                output={},
                summary="replan 실패: context에 claim 없음",
                success=False,
                error="claim_unavailable",
            )

        # ── 호출 횟수 제한 ─────────────────────────────────────────
        count = _read_replan_count(workspace, claim_id)
        if count >= _REPLAN_MAX_PER_CLAIM:
            logger.warning(
                f"[replan] {claim_id}: max {_REPLAN_MAX_PER_CLAIM}회 도달 — "
                f"replan 거부"
            )
            return ToolResult(
                output={"replan_count": count, "max": _REPLAN_MAX_PER_CLAIM},
                summary=(
                    f"replan 거부: 이미 {count}회 시도. 더 이상 plan 재작성 없이 "
                    f"현재까지의 evidence로 finish 결정 필요."
                ),
                success=False,
                error="replan_limit_exceeded",
            )

        # ── observation 컨텍스트 수집 ──────────────────────────────
        obs_summary = _collect_observation_summary(workspace, claim_id)
        logger.info(
            f"[replan] {claim_id}: 호출 #{count + 1}/{_REPLAN_MAX_PER_CLAIM}, "
            f"reason={reason[:120]!r}, observations={len(obs_summary)}건"
        )

        original_plan = getattr(context, "current_plan", None)
        config = getattr(context, "config", None) or {}

        # ── Step 1: schema regenerate ──────────────────────────────
        # 원래 schema가 잘못 분류된 경우(예: '증가 수'를 base로) — 표 row sample
        # 보여주고 LLM에게 value_role/prev_time 재분류 받기.
        from structverify.detection.schema_inductor import regenerate_schema
        old_schema_dict: dict | None = None
        try:
            _orig = getattr(claim, "schema", None)
            if _orig is not None and hasattr(_orig, "model_dump"):
                old_schema_dict = _orig.model_dump(mode="json")
            elif isinstance(_orig, dict):
                old_schema_dict = dict(_orig)
        except Exception:
            old_schema_dict = None

        claim_text = getattr(claim, "claim_text", "") or ""
        new_schema_dict: dict | None = None
        try:
            new_schema_dict = await regenerate_schema(
                claim_text=str(claim_text),
                original_schema=old_schema_dict,
                observations=obs_summary,
                config=config,
            )
        except Exception as e:
            logger.warning(f"[replan] {claim_id}: regenerate_schema 호출 실패: {e}")
            new_schema_dict = None

        # ── Step 2: schema 변경됐는지 확인 (no-op 거부) ─────────────
        # value_role이 바뀌면 *근본적*으로 다른 검증 — replan 진행.
        # value_role 그대로면 의미 없는 replan → 거부.
        if new_schema_dict is None:
            _write_replan_count(workspace, claim_id, count + 1)
            return ToolResult(
                output={"replan_count": count + 1},
                summary=(
                    "replan 실패: schema 재분류 실패. observation에 표 데이터가 "
                    "충분하지 않거나 LLM이 새 schema 생성 못 함. "
                    "남은 evidence로 finish 결정 권장."
                ),
                success=False,
                error="regenerate_schema_failed",
            )

        old_role = (old_schema_dict or {}).get("value_role")
        new_role = new_schema_dict.get("value_role")
        if old_role == new_role and (old_schema_dict or {}).get("prev_time_period") == new_schema_dict.get("prev_time_period"):
            # schema 의미 변화 없음 — replan 의미 없음
            _write_replan_count(workspace, claim_id, count + 1)
            logger.warning(
                f"[replan] {claim_id}: schema 변화 없음 "
                f"(value_role={old_role!r} 그대로, prev_time도 동일) — no-op 거부"
            )
            return ToolResult(
                output={
                    "replan_count": count + 1,
                    "old_value_role": old_role,
                    "new_value_role": new_role,
                },
                summary=(
                    f"replan 거부: schema 의미 변화 없음 (value_role={old_role!r} 유지). "
                    f"이건 같은 plan 반복 시도일 뿐 — *구조적* 재분류가 일어나야 replan. "
                    f"현재 evidence로 finish 결정 필요."
                ),
                success=False,
                error="schema_unchanged",
            )

        # ── Step 3: claim.schema 업데이트 ──────────────────────────
        # ClaimSchema model_copy로 새 schema 적용 → claim 객체 동기화.
        try:
            from structverify.core.schemas import ClaimSchema
            _orig_schema = getattr(claim, "schema", None)
            if _orig_schema is not None and hasattr(_orig_schema, "model_copy"):
                # 새 dict로 update
                _update = {k: v for k, v in new_schema_dict.items() if v is not None}
                _new_schema_obj = _orig_schema.model_copy(update=_update)
            else:
                # 원래 schema 없으면 새로 생성
                _new_schema_obj = ClaimSchema(**new_schema_dict)
            # claim 객체에 직접 setattr — pydantic 모델이라도 가능 (frozen 아니면)
            try:
                claim.schema = _new_schema_obj
            except Exception:
                # frozen 모델이면 새 claim 생성 시도
                if hasattr(claim, "model_copy"):
                    _new_claim = claim.model_copy(update={"schema": _new_schema_obj})
                    # context.claim 도 갱신 (이후 iter가 새 claim 봐야 함)
                    context.claim = _new_claim
                    claim = _new_claim
        except Exception as e:
            logger.warning(f"[replan] {claim_id}: claim.schema 업데이트 실패: {e}")
            _write_replan_count(workspace, claim_id, count + 1)
            return ToolResult(
                output={"replan_count": count + 1},
                summary=f"replan 실패: schema 업데이트 오류 — {type(e).__name__}: {e}",
                success=False,
                error=f"schema_update_failed: {e}",
            )

        logger.info(
            f"[replan] {claim_id}: schema 업데이트 완료 — "
            f"value_role: {old_role!r} → {new_role!r}, "
            f"prev_time_period: {(old_schema_dict or {}).get('prev_time_period')!r} → "
            f"{new_schema_dict.get('prev_time_period')!r}"
        )

        # ── Step 4: planner.plan 재호출 — 새 schema로 자연스러운 plan ──
        # value_role이 바뀌었으면 planner의 결정적 보정([planner.py:597-611])이
        # 자동으로 claim_type을 새 value_role에 맞게 설정.
        try:
            from structverify.agent.planner import Planner, PlannerConfig
            from structverify.utils.llm_client import LLMClient
            _llm = LLMClient(config=(config or {}).get("llm") or {})
            async def _llm_call(prompt: str) -> str:
                return await _llm.generate(
                    prompt=prompt,
                    system_prompt="당신은 통계 검증 planner 입니다. JSON으로만 응답.",
                    model_tier="heavy",
                )
            planner = Planner(llm_call=_llm_call, config=PlannerConfig())
            new_plan = await planner.plan(claim=claim)
        except Exception as e:
            logger.warning(f"[replan] {claim_id}: planner.plan 재호출 실패: {e}")
            _write_replan_count(workspace, claim_id, count + 1)
            return ToolResult(
                output={"replan_count": count + 1},
                summary=f"replan 실패 (schema는 갱신됨, plan 재생성 실패): {type(e).__name__}: {e}",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        # 카운트 증가 (전체 성공 시점)
        _write_replan_count(workspace, claim_id, count + 1)

        if new_plan is None:
            return ToolResult(
                output={"replan_count": count + 1},
                summary="replan 실패: planner.plan이 None 반환",
                success=False,
                error="planner_returned_none",
            )

        logger.info(
            f"[replan] {claim_id}: 새 plan 생성 완료 — "
            f"type={new_plan.claim_type.value}, "
            f"steps={len(new_plan.initial_steps)}, "
            f"formula={new_plan.calculation_formula!r}"
        )

        # plan을 dict로 직렬화해 output에 실어 보냄. loop이 받아서 교체.
        try:
            new_plan_dict = new_plan.model_dump(mode="json")
        except Exception:
            new_plan_dict = {"_serialize_error": True}

        return ToolResult(
            output={
                "new_plan": new_plan_dict,
                "new_schema": new_schema_dict,
                "old_value_role": old_role,
                "new_value_role": new_role,
                "replan_count": count + 1,
                "reason": reason,
            },
            summary=(
                f"replan 성공 (#{count + 1}): "
                f"value_role {old_role!r} → {new_role!r}, "
                f"claim_type={new_plan.claim_type.value}, "
                f"steps={len(new_plan.initial_steps)}, "
                f"formula={new_plan.calculation_formula!r}. "
                f"이후 iter는 *새 schema + 새 plan*으로 진행 — 필요한 시점들을 fetch."
            ),
            success=True,
        )
