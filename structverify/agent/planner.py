"""
agent/planner.py — 전략 수립 및 롤백 방향 결정 (Planner)

에이전틱 루프에서 LLM이 두 가지 판단을 내린다:
  1) 스텝 진입 전 실행 전략 수립 (plan_step)
  2) Critic이 ROLLBACK 결정 시 복구 방향 결정 (plan_rollback)

기존 레이어의 LLM 호출과 독립적으로 동작한다.
LLMClient는 config["llm"]에서 가져온다.

- 담당자: 신준수 [agent/v1]
"""
# 수정자: 신준수
# 버전: agent/v1
# 수정 날짜: 2026-05-15
# 수정 내용: 에이전틱 리팩토링 - Planner 전략/롤백 LLM 판단 모듈 신규

# [DONE] PLAN_STEP_PROMPT 정의
# [DONE] PLAN_ROLLBACK_PROMPT 정의
# [DONE] plan_step() 구현
# [DONE] plan_rollback() 구현
# [DONE] LLM 응답 파싱 실패 시 안전 폴백 처리
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

if TYPE_CHECKING:
    from structverify.agent.context import RunContext

logger = get_logger(__name__)


# ── 프롬프트 ──────────────────────────────────────────────────────────────────

PLAN_STEP_PROMPT = """당신은 팩트체크 파이프라인의 전략 플래너입니다.
다음 스텝을 실행하기 전에 최적의 실행 전략을 결정하세요.

[검증 대상 주장]
{claim_text}

[문서 도메인]
{domain}

[현재 스텝]
{step} ({step_desc})

[이전 시도 이력]
{history}

[현재 스키마 정보]
{schema_summary}

위 정보를 바탕으로 이 스텝에서 사용할 최적 전략을 JSON으로 답하세요.
전략이 없거나 기본 실행으로 충분하면 hint를 빈 문자열로 두세요.

JSON:
{{"strategy": "전략 한 줄 설명", "hint": "레이어 함수에 전달할 힌트 (빈 문자열 허용)"}}
"""

PLAN_ROLLBACK_PROMPT = """당신은 팩트체크 파이프라인의 복구 플래너입니다.
스텝 실패 원인을 분석하고 어떻게 복구할지 결정하세요.

[검증 대상 주장]
{claim_text}

[문서 도메인]
{domain}

[실패한 스텝]
{failed_step} ({failed_step_desc})

[실패 원인 / Critic 판단]
{failed_reason}

[현재 스키마 정보]
{schema_summary}

[지금까지 롤백 이력]
{rollback_history}

[남은 시도 횟수]
{remaining_attempts}회

복구 방향을 JSON으로 결정하세요.

판단 기준:
- indicator가 너무 구체적이면 → rollback_to=5, hint에 "indicator 단순화" 지시
- time_period 오류 → rollback_to=5, hint에 "time_period 재확인" 지시
- KOSIS에 없는 데이터면 → give_up=true
- 남은 시도 횟수가 0이면 → give_up=true

JSON:
{{
  "rollback_to": 5,
  "reason": "롤백 이유 한 줄",
  "hint": "Step 5 재시도 시 주입할 힌트 (indicator/source_phrase 타겟 명시 권장)",
  "give_up": false
}}
"""

# 스텝 설명 (프롬프트 주입용)
_STEP_DESC = {
    5: "schema 유도 — indicator/value/unit/time_period 추출",
    7: "KOSIS 검색 — 공식 통계 수치 조회",
    8: "팩트 판별 — claim 수치 vs KOSIS 수치 비교",
    9: "설명 생성 — 판정 결과 자연어 변환",
}


# ── 메인 함수 ─────────────────────────────────────────────────────────────────

async def plan_step(
    step: int,
    ctx: "RunContext",
    config: dict,
) -> dict:
    """
    스텝 진입 전 LLM에게 실행 전략 힌트를 요청한다.

    Step 5, 7에서만 호출. 나머지 스텝은 전략 불필요.
    LLM 호출 실패 시 빈 전략({})으로 폴백 — 파이프라인 중단 없음.

    Returns:
        {"strategy": str, "hint": str}
    """
    if step not in (5, 7):
        return {}

    llm = LLMClient(config=config.get("llm", {}))
    prompt = PLAN_STEP_PROMPT.format(
        claim_text=ctx.claim.claim_text,
        domain=config.get("detected_domain", "general"),
        step=step,
        step_desc=_STEP_DESC.get(step, ""),
        history=_format_rollback_history(ctx),
        schema_summary=_format_schema(ctx),
    )

    try:
        result = await llm.generate_json(
            prompt=prompt,
            system_prompt="팩트체크 전략 플래너. JSON으로만 답하세요.",
            model_tier="light",  # 경량 모델 — 전략 수립은 빠르게
        )
        strategy = str(result.get("strategy", ""))
        hint = str(result.get("hint", ""))
        logger.info(
            f"[Planner] plan_step step={step} claim={ctx.claim.sent_id} "
            f"strategy={strategy!r} hint={hint!r}"
        )
        return {"strategy": strategy, "hint": hint}

    except Exception as e:
        logger.warning(f"[Planner] plan_step LLM 호출 실패 (폴백): {e}")
        return {}


async def plan_rollback(
    ctx: "RunContext",
    failed_step: int,
    config: dict,
) -> dict:
    """
    Critic이 ROLLBACK을 결정했을 때 복구 방향을 LLM에게 결정한다.

    Returns:
        {
            "rollback_to": int,   # 돌아갈 스텝 번호
            "reason": str,        # 롤백 이유
            "hint": str,          # 롤백 스텝 재실행 시 주입할 힌트
            "give_up": bool,      # True면 이 claim 검증 포기
        }
    """
    # 남은 시도 횟수가 0이면 LLM 없이 즉시 GIVE_UP
    remaining = ctx.max_attempts - ctx.attempt_count
    if remaining <= 0:
        logger.info(
            f"[Planner] plan_rollback — 시도 횟수 소진 "
            f"(attempt={ctx.attempt_count}/{ctx.max_attempts}), GIVE_UP"
        )
        return {
            "rollback_to": failed_step,
            "reason": "최대 시도 횟수 초과",
            "hint": "",
            "give_up": True,
        }

    # 마지막 Critic 판단 실패 원인 (snapshot에서 추출)
    last_snapshot = ctx.last_snapshot(failed_step)
    failed_reason = (
        last_snapshot.failed_reason if last_snapshot else "알 수 없음"
    ) or "알 수 없음"

    llm = LLMClient(config=config.get("llm", {}))
    prompt = PLAN_ROLLBACK_PROMPT.format(
        claim_text=ctx.claim.claim_text,
        domain=config.get("detected_domain", "general"),
        failed_step=failed_step,
        failed_step_desc=_STEP_DESC.get(failed_step, ""),
        failed_reason=failed_reason,
        schema_summary=_format_schema(ctx),
        rollback_history=_format_rollback_history(ctx),
        remaining_attempts=remaining,
    )

    try:
        result = await llm.generate_json(
            prompt=prompt,
            system_prompt="팩트체크 복구 플래너. JSON으로만 답하세요.",
            model_tier="light",
        )

        rollback_to = int(result.get("rollback_to", 5))
        reason      = str(result.get("reason", ""))
        hint        = str(result.get("hint", ""))
        give_up     = bool(result.get("give_up", False))

        # 롤백 대상 스텝 안전 범위 확인 (5 이상, failed_step 미만)
        if rollback_to < 5 or rollback_to >= failed_step:
            logger.warning(
                f"[Planner] rollback_to={rollback_to} 범위 오류 → 5로 보정"
            )
            rollback_to = 5

        logger.info(
            f"[Planner] plan_rollback claim={ctx.claim.sent_id} "
            f"failed={failed_step} → rollback_to={rollback_to} "
            f"give_up={give_up} reason={reason!r}"
        )
        return {
            "rollback_to": rollback_to,
            "reason": reason,
            "hint": hint,
            "give_up": give_up,
        }

    except Exception as e:
        logger.warning(f"[Planner] plan_rollback LLM 호출 실패 (안전 폴백): {e}")
        # 목적: LLM 실패 시 기본적으로 Step 5로 롤백
        return {
            "rollback_to": 5,
            "reason": f"Planner LLM 오류 — 기본 롤백: {e}",
            "hint": "",
            "give_up": False,
        }


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────

def _format_schema(ctx: "RunContext") -> str:
    """현재 claim.schema를 프롬프트 주입용 문자열로 변환."""
    s = ctx.claim.schema
    if s is None:
        return "schema 없음"
    parts = []
    if s.indicator:     parts.append(f"indicator={s.indicator!r}")
    if s.value:         parts.append(f"value={s.value}")
    if s.unit:          parts.append(f"unit={s.unit!r}")
    if s.time_period:   parts.append(f"time_period={s.time_period!r}")
    if s.population:    parts.append(f"population={s.population!r}")
    if s.parent_path:   parts.append(f"parent_path={s.parent_path!r}")
    return ", ".join(parts) if parts else "schema 필드 없음"


def _format_rollback_history(ctx: "RunContext") -> str:
    """롤백 이력을 프롬프트 주입용 문자열로 변환."""
    if not ctx.rollback_log:
        return "없음 (첫 시도)"
    lines = []
    for i, log in enumerate(ctx.rollback_log, 1):
        lines.append(
            f"{i}. rollback_to={log.get('rollback_to')} "
            f"reason={log.get('reason', '')!r} "
            f"hint={log.get('hint', '')!r}"
        )
    return "\n".join(lines)
