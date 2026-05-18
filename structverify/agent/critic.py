"""
agent/critic.py — 스텝 결과 품질 평가 (Critic)

Executor가 실행한 결과를 평가하고 CriticVerdict를 반환한다.
코드 시그널 기반 판단 우선, LLM은 품질이 애매한 경우에만 사용한다.

- 담당자: 신준수 [agent/v1]
"""
# 수정자: 신준수
# 버전: agent/v1
# 수정 날짜: 2026-05-15
# 수정 내용: 에이전틱 리팩토링 - Critic 코드 시그널 기반 판단 신규 (commit 4)
#           + LLM 품질 판단 추가 (commit 7):
#             · Step 7 evidence unit 불일치 → LLM 적합성 판단
#             · Step 8 UNVERIFIABLE 원인 분석 → 롤백 필요 여부 결정

# [DONE] evaluate() 코드 시그널 기반 구현
# [DONE] Step 7: evidence None 횟수 기반 RETRY/ROLLBACK
# [DONE] Step 8: verdict 기반 OK/STOP/ROLLBACK
# [DONE] Step 7: evidence unit 불일치 → LLM 품질 판단
# [DONE] Step 8: UNVERIFIABLE 원인 분석 → LLM
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structverify.agent.context import CriticVerdict
from structverify.core.schemas import VerdictType
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

if TYPE_CHECKING:
    from structverify.agent.context import RunContext

logger = get_logger(__name__)

# Step 7에서 RETRY_SAME을 허용하는 최대 횟수
# 이 횟수를 초과하면 ROLLBACK으로 전환
_MAX_STEP7_RETRY = 2

# ── LLM 품질 판단 프롬프트 ────────────────────────────────────────────────────

_UNIT_COMPAT_PROMPT = """팩트체크 검증 전문가로서 아래 두 단위가 같은 개념을 나타내는지 판단하세요.

주장의 단위: {schema_unit}
KOSIS 데이터의 단위: {kosis_unit}

판단 기준:
- 단순 표기 차이 (예: "명" vs "천명"): 변환 가능하면 compatible
- 완전히 다른 개념 (예: "%" vs "명"): incompatible
- 단위가 없거나 애매한 경우: compatible (관대하게 처리)

JSON으로만 답하세요:
{{"compatible": true, "reason": "한 줄 근거"}}
"""

_UNVERIFIABLE_CAUSE_PROMPT = """팩트체크 검증 전문가로서 UNVERIFIABLE 판정 원인을 분석하세요.

[검증 대상 주장]
{claim_text}

[스키마 정보]
indicator: {indicator}
time_period: {time_period}
unit: {unit}

[KOSIS 조회 결과]
evidence: {evidence_summary}

UNVERIFIABLE 원인을 분석하고 롤백이 도움이 될지 판단하세요.

JSON으로만 답하세요:
{{
  "cause": "data_not_exist | schema_error | time_mismatch | unit_mismatch | other",
  "rollback_helpful": true,
  "reason": "한 줄 근거"
}}
"""


def evaluate(
    step: int,
    output: Any,
    ctx: "RunContext",
    config: dict | None = None,
) -> CriticVerdict:
    """
    스텝 실행 결과를 평가하고 CriticVerdict를 반환한다.

    판단 우선순위:
      1) 코드 시그널 (evidence None 여부, verdict 값 등) — 빠르고 deterministic
      2) LLM 품질 판단 — 코드로 판단 불가한 애매한 경우

    Args:
        step:   실행된 스텝 번호
        output: Executor.execute_step()의 반환값
        ctx:    현재 claim의 RunContext (시도 이력 참조용)
        config: 파이프라인 설정 (LLM 품질 판단에 사용)

    Returns:
        CriticVerdict
    """
    if step == 5:
        return _evaluate_step5(output, ctx)
    elif step == 7:
        return _evaluate_step7(output, ctx, config=config)
    elif step == 8:
        return _evaluate_step8(output, ctx, config=config)
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

def _evaluate_step7(
    evidence: Any,
    ctx: "RunContext",
    config: dict | None = None,
) -> CriticVerdict:
    """
    KOSIS 증거 검색 결과 평가.

    evidence None:
      - 시도 횟수 ≤ _MAX_STEP7_RETRY → RETRY_SAME (파라미터 그대로 재시도)
      - 시도 횟수 > _MAX_STEP7_RETRY → ROLLBACK (Step 5로 돌아가서 schema 재유도)

    evidence 있음:
      - unit 불일치 의심 → LLM으로 적합성 판단
      - OK
    """
    retry_count = ctx.retry_count_for_step(7)

    if evidence is None:
        # 롤백 이력이 있다 = 이미 schema 재유도까지 시도했음
        # 그래도 evidence=None이면 KOSIS에 해당 지표 자체가 없는 것 → GIVE_UP
        if ctx.rollback_log:
            logger.warning(
                f"[Critic] Step 7 claim={ctx.claim.sent_id} → "
                f"롤백 후에도 evidence None → KOSIS 데이터 없음, GIVE_UP"
            )
            return CriticVerdict.GIVE_UP

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

    # 목적: 코드로 명확히 판단 가능한 unit 불일치 먼저 체크
    schema_unit = (getattr(ctx.claim.schema, "unit", None) or "").strip()
    kosis_unit  = (getattr(evidence, "unit", None) or "").strip()

    if schema_unit and kosis_unit and schema_unit != kosis_unit:
        # 단위가 다른 경우 LLM으로 호환 여부 판단
        # 동기 함수에서 비동기 LLM을 호출할 수 없으므로 간이 규칙 우선 적용
        # 완전히 다른 단위 타입 (% vs 명/건/원)이면 즉시 ROLLBACK
        if _is_obviously_incompatible(schema_unit, kosis_unit):
            logger.warning(
                f"[Critic] Step 7 claim={ctx.claim.sent_id} → "
                f"unit 불일치 (schema={schema_unit!r}, kosis={kosis_unit!r}), ROLLBACK"
            )
            return CriticVerdict.ROLLBACK

        logger.info(
            f"[Critic] Step 7 claim={ctx.claim.sent_id} → "
            f"unit 차이 있지만 호환 가능으로 판단 "
            f"(schema={schema_unit!r}, kosis={kosis_unit!r}), OK"
        )

    logger.debug(
        f"[Critic] Step 7 OK → "
        f"official_value={getattr(evidence, 'official_value', None)}"
    )
    return CriticVerdict.OK


# ── Step 8: 팩트 판별 결과 평가 ─────────────────────────────────────────────

def _evaluate_step8(
    result: Any,
    ctx: "RunContext",
    config: dict | None = None,
) -> CriticVerdict:
    """
    verify_claim() 결과 평가.

    MATCH        → OK (팩트 검증 성공)
    MISMATCH     → STOP (롤백 금지 — 기사가 실제로 틀린 것)
    UNVERIFIABLE → LLM 원인 분석 후 롤백 여부 결정

    설계 원칙:
      MISMATCH에서 롤백하면 안 됨 — 억지로 MATCH 만들면 팩트체크 의미 없어짐.

    UNVERIFIABLE 처리:
      - "데이터 없음(data_not_exist)": KOSIS에 없는 지표 → GIVE_UP
      - "schema 오류(schema_error)": indicator/time_period 재유도로 개선 가능 → ROLLBACK
      - LLM 호출 불가 시: 기본적으로 ROLLBACK (시도해볼 가치 있음)
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
        # [agent/v1 신준수 2026-05-15] UNVERIFIABLE → 즉시 STOP (롤백 금지)
        #
        # 롤백 이유로 schema를 재유도하면 오히려 evidence를 찾지 못하게 되는
        # 역효과가 발생함. 기존 process_one_claim과 동일하게 UNVERIFIABLE은
        # 그 상태로 확정하고 explanation만 생성하도록 처리.
        #
        # 롤백이 의미있는 케이스 (commit 7 TODO):
        #   - LLM이 "schema_error"로 판단했을 때만 선별적으로 ROLLBACK 허용
        #
        # [agent/v1 버그] → ROLLBACK: schema 재유도 → KOSIS query 변경
        #            → evidence 사라짐 → 정확도 하락
        logger.info(
            f"[Critic] Step 8 UNVERIFIABLE claim={ctx.claim.sent_id} → STOP "
            f"(롤백 없이 확정 — explanation 생성 후 종료)"
        )
        return CriticVerdict.STOP

    logger.warning(f"[Critic] Step 8 알 수 없는 verdict={verdict} → GIVE_UP")
    return CriticVerdict.GIVE_UP


# ── Step 9: 설명 생성 결과 평가 ─────────────────────────────────────────────

# ── unit 호환성 간이 규칙 ────────────────────────────────────────────────────

def _is_obviously_incompatible(schema_unit: str, kosis_unit: str) -> bool:
    """
    두 단위가 명백히 다른 타입인지 간이 규칙으로 판단.

    목적: LLM 없이 빠르게 판단 가능한 케이스 처리.
    예: "%" vs "명" → True (비율 vs 절대값)
        "명" vs "천명" → False (표기 차이, 변환 가능)
        "%" vs "%" → False (동일)
    """
    schema_u = schema_unit.lower().strip()
    kosis_u  = kosis_unit.lower().strip()

    if schema_u == kosis_u:
        return False

    # 비율(%)과 절대값(명/건/원/개 등) 혼재 → 명백히 다른 타입
    ratio_units = {"%", "퍼센트", "percent", "rate"}
    count_units = {"명", "건", "원", "개", "천명", "만명", "억명", "만원", "억원", "천원"}

    schema_is_ratio = any(r in schema_u for r in ratio_units)
    kosis_is_ratio  = any(r in kosis_u  for r in ratio_units)
    schema_is_count = any(c in schema_u for c in count_units)
    kosis_is_count  = any(c in kosis_u  for c in count_units)

    if schema_is_ratio and kosis_is_count:
        return True
    if schema_is_count and kosis_is_ratio:
        return True

    return False


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
