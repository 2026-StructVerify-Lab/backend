"""
agent/critic.py — 스텝 결과 품질 평가 (Critic)

Executor가 실행한 결과를 평가하고 CriticVerdict를 반환한다.
코드 시그널 기반 판단 우선, LLM은 품질이 애매한 경우에만 사용한다.

- 담당자: 신준수
"""
# 수정자: 신준수
# 수정 날짜: 2026-05-15
# 수정 내용: 에이전틱 리팩토링 - Critic 코드 시그널 기반 판단 신규

# [DONE] evaluate() 코드 시그널 기반 구현
# [DONE] Step 7: evidence None 횟수 기반 RETRY/ROLLBACK
# [DONE] Step 8: verdict 기반 OK/STOP/ROLLBACK
# [TODO] Step 7: evidence unit 불일치 → LLM 품질 판단 (commit 7에서 추가)
# [TODO] Step 8: UNVERIFIABLE 원인 분석 → LLM (commit 7에서 추가)
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structverify.agent.context import CriticVerdict
from structverify.core.schemas import VerdictType
from structverify.utils.logger import get_logger

if TYPE_CHECKING:
    from structverify.agent.context import RunContext

logger = get_logger(__name__)

# Step 7에서 RETRY_SAME을 허용하는 최대 횟수
# 이 횟수를 초과하면 ROLLBACK으로 전환
_MAX_STEP7_RETRY = 2


def evaluate(step: int, output: Any, ctx: "RunContext") -> CriticVerdict:
    """
    스텝 실행 결과를 평가하고 CriticVerdict를 반환한다.

    판단 우선순위:
      1) 코드 시그널 (evidence None 여부, verdict 값 등) — 빠르고 deterministic
      2) LLM 품질 판단 — 코드로 판단 불가한 애매한 경우 (TODO: commit 7)

    Args:
        step:   실행된 스텝 번호
        output: Executor.execute_step()의 반환값
        ctx:    현재 claim의 RunContext (시도 이력 참조용)

    Returns:
        CriticVerdict
    """
    if step == 5:
        return _evaluate_step5(output, ctx)
    elif step == 7:
        return _evaluate_step7(output, ctx)
    elif step == 8:
        return _evaluate_step8(output, ctx)
    elif step == 9:
        return _evaluate_step9(output, ctx)
    else:
        logger.warning(f"[Critic] 알 수 없는 스텝 {step} → OK 처리")
        return CriticVerdict.OK


# ── Step 5: schema 유도 결과 평가 ───────────────────────────────────────────

def _evaluate_step5(schema: Any, ctx: "RunContext") -> CriticVerdict:
    """
    schema 유도 결과 평가.

    schema가 None이면 indicator 추출 실패 → ROLLBACK 불가하므로 GIVE_UP.
    indicator가 너무 짧으면 품질 불량 → ROLLBACK (Step 5 재시도 hint 변경).
    """
    if schema is None:
        logger.warning(
            f"[Critic] Step 5 claim={ctx.claim.sent_id} → schema None, GIVE_UP"
        )
        return CriticVerdict.GIVE_UP

    indicator = getattr(schema, "indicator", None) or ""
    if len(indicator.strip()) < 2:
        logger.warning(
            f"[Critic] Step 5 claim={ctx.claim.sent_id} → indicator 너무 짧음, GIVE_UP"
        )
        return CriticVerdict.GIVE_UP

    logger.debug(f"[Critic] Step 5 OK → indicator={indicator}")
    return CriticVerdict.OK


# ── Step 7: KOSIS 증거 검색 결과 평가 ───────────────────────────────────────

def _evaluate_step7(evidence: Any, ctx: "RunContext") -> CriticVerdict:
    """
    KOSIS 증거 검색 결과 평가.

    evidence None:
      - 시도 횟수 ≤ _MAX_STEP7_RETRY → RETRY_SAME (파라미터 그대로 재시도)
      - 시도 횟수 > _MAX_STEP7_RETRY → ROLLBACK (Step 5로 돌아가서 schema 재유도)

    evidence 있음:
      - OK (단위 불일치 등 LLM 품질 판단은 TODO: commit 7에서 추가)
    """
    retry_count = ctx.retry_count_for_step(7)

    if evidence is None:
        if retry_count < _MAX_STEP7_RETRY:
            logger.info(
                f"[Critic] Step 7 claim={ctx.claim.sent_id} → evidence None "
                f"(retry {retry_count + 1}/{_MAX_STEP7_RETRY}), RETRY_SAME"
            )
            return CriticVerdict.RETRY_SAME
        else:
            logger.warning(
                f"[Critic] Step 7 claim={ctx.claim.sent_id} → evidence None "
                f"{retry_count + 1}회 초과, ROLLBACK"
            )
            return CriticVerdict.ROLLBACK

    # TODO [신준수 - commit 7]: evidence unit 불일치 케이스 → LLM 품질 판단
    # schema_unit = getattr(ctx.claim.schema, "unit", None)
    # kosis_unit = getattr(evidence, "unit", None)
    # if schema_unit and kosis_unit and not _units_compatible(schema_unit, kosis_unit):
    #     return await _llm_quality_check(evidence, ctx)

    logger.debug(
        f"[Critic] Step 7 OK → "
        f"official_value={getattr(evidence, 'official_value', None)}"
    )
    return CriticVerdict.OK


# ── Step 8: 팩트 판별 결과 평가 ─────────────────────────────────────────────

def _evaluate_step8(result: Any, ctx: "RunContext") -> CriticVerdict:
    """
    verify_claim() 결과 평가.

    MATCH        → OK (팩트 검증 성공)
    MISMATCH     → STOP (롤백 금지 — 기사가 실제로 틀린 것)
    UNVERIFIABLE → ROLLBACK (데이터 없거나 schema 재유도로 개선 가능)

    설계 원칙:
      MISMATCH에서 롤백하면 안 됨 — 억지로 MATCH 만들면 팩트체크 의미 없어짐.
    """
    if result is None:
        logger.warning(f"[Critic] Step 8 claim={ctx.claim.sent_id} → result None, GIVE_UP")
        return CriticVerdict.GIVE_UP

    verdict = getattr(result, "verdict", None)

    if verdict == VerdictType.MATCH:
        logger.debug(f"[Critic] Step 8 MATCH → OK")
        return CriticVerdict.OK

    elif verdict == VerdictType.MISMATCH:
        # 목적: MISMATCH는 롤백하지 않음 — 기사 수치가 실제로 틀린 것
        logger.info(
            f"[Critic] Step 8 MISMATCH claim={ctx.claim.sent_id} → STOP (롤백 금지)"
        )
        return CriticVerdict.STOP

    elif verdict == VerdictType.UNVERIFIABLE:
        # TODO [신준수 - commit 7]: UNVERIFIABLE 원인 LLM 분석
        # → "데이터 없음" vs "schema 오류"를 구분하여 롤백 여부 결정
        logger.info(
            f"[Critic] Step 8 UNVERIFIABLE claim={ctx.claim.sent_id} → ROLLBACK"
        )
        return CriticVerdict.ROLLBACK

    logger.warning(f"[Critic] Step 8 알 수 없는 verdict={verdict} → GIVE_UP")
    return CriticVerdict.GIVE_UP


# ── Step 9: 설명 생성 결과 평가 ─────────────────────────────────────────────

def _evaluate_step9(explanation: Any, ctx: "RunContext") -> CriticVerdict:
    """
    설명 생성 결과 평가.

    설명이 없거나 너무 짧으면 경고 로그만 남기고 OK 처리.
    (설명 실패가 검증 결과를 뒤집을 이유는 없음)
    """
    if not explanation or len(str(explanation).strip()) < 10:
        logger.warning(
            f"[Critic] Step 9 claim={ctx.claim.sent_id} → 설명 짧음/None, OK로 처리"
        )
    return CriticVerdict.OK
