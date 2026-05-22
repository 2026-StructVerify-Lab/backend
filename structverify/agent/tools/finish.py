"""
structverify.agent.tools.finish — 종료 Tool.

Agent가 *충분히 검증했거나 더 시도해도 안 된다고 판단*했을 때 호출.

이 Tool을 호출하면:
  - workspace.write_verdict()로 verdict.json 저장
  - memory.md에 "## Final Verdict" 섹션 추가
  - Loop은 *이 Tool 호출 후 즉시 종료*

호출 시 LLM이 결정해야 할 것:
  - verdict: "match" | "mismatch" | "partial" | "unverifiable"
  - confidence: 0.0 ~ 1.0
  - explanation: 사람-읽기 자연어 (최종 출력)
  - data_points: 모은 데이터 점들 (검산용)
"""
from __future__ import annotations

from structverify.utils.logger import get_logger
from datetime import datetime, timezone
from typing import Any

from ..schemas import ActionType, AgentVerdict, DataPointSpec, StopReason, VerdictType
from ..memory import append_final
from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


_VALID_VERDICTS = {v.value for v in VerdictType}


def _has_successful_fetch_evidence(workspace, claim_id) -> bool:
    """이 claim에 대해 fetch_evidence가 한 번이라도 success로 끝났는지.

    LLM이 evidence 한 번도 못 받았는데 match/mismatch로 finish 호출하는
    hallucination을 차단하기 위한 가드. (1-2 A안)
    """
    try:
        names = workspace.list_observations(claim_id)
    except Exception as e:
        logger.debug(f"[finish] list_observations 실패: {e}")
        return False
    for name in names:
        if "fetch" not in name.lower():
            continue
        data = workspace.read_observation(claim_id, name)
        if not isinstance(data, dict):
            continue
        # 표준 observation 형식
        if data.get("action") == "fetch_evidence" and data.get("success") is True:
            return True
        # 일부 raw 저장 형식 — evidence.value가 있으면 성공으로 간주
        ev = (data.get("output") or {}).get("evidence") or data.get("evidence") or {}
        if isinstance(ev, dict) and ev.get("value") is not None:
            return True
    return False


@register_tool(ActionType.FINISH)
class FinishTool(ToolBase):
    """검증 종료 + Verdict 확정.

    이 Tool 호출 후 Loop은 *즉시 종료*. 다음 iteration 없음.

    호출 시점:
      - 모든 데이터 점 확보 + 계산 완료 → MATCH / MISMATCH
      - 일부만 확보됐는데 충분히 정황 파악 → PARTIAL
      - 시도했지만 데이터 못 찾음 → UNVERIFIABLE
    """

    name = ActionType.FINISH
    description = (
        "검증 종료. 모든 데이터 모았거나 더 시도해도 안 될 때 호출. "
        "verdict 결정 + 사용자에게 보일 explanation 작성. "
        "이 Tool 호출 후 loop이 종료되므로 *마지막 결정* 신중히."
    )
    input_schema = {
        "verdict": (
            "판정. 'match' (일치) | 'mismatch' (불일치, 시점/단위 같지만 값 다름) | "
            "'partial' (일부 검증) | 'unverifiable' (검증 불가능)"
        ),
        "confidence": "신뢰도 (0.0~1.0). match면 보통 0.9+, partial은 0.5~0.8, unverifiable은 0.2~0.5",
        "explanation": "사용자에게 보일 설명 (자연어 2-4문장, KOSIS 출처 포함)",
        "data_points": "확보한 데이터 점들 (선택). [{indicator, time, resolved_value, source}, ...]",
    }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        # 입력 파싱
        verdict_raw = (input_data.get("verdict") or "").strip().lower()
        # [2026-05-21] reflect LLM이 verdict 자리에 claim_type(comparison/absolute/...)을
        # 잘못 박는 헛돌이 차단 — reject(success=False)하면 LLM이 같은 실수 반복.
        # 대신 unverifiable로 강등하고 finish는 *성공*시켜 loop을 종료 → 직후
        # N 패치가 schema.value vs evidence 객관 비교해 MATCH/MISMATCH로 정정 가능.
        # 22:54~22:55 로그: 661202e8 claim이 iter 4, 6 두 번 verdict='comparison' 실패
        # → iter 7 unverifiable로 끝난 케이스 해결.
        _claim_type_misvalues = {
            "comparison", "absolute", "growth_rate", "difference",
            "ranking", "diff", "absolute_value", "compare",
        }
        if verdict_raw not in _VALID_VERDICTS:
            if verdict_raw in _claim_type_misvalues:
                logger.warning(
                    f"[finish] {context.claim_id}: verdict={verdict_raw!r}는 "
                    f"claim_type 값 — verdict 필드에 잘못 박힘. unverifiable로 강등 "
                    f"(loop 종료 후 합성 verdict가 객관 비교로 정정 가능)"
                )
                verdict_raw = "unverifiable"
            else:
                return ToolResult(
                    output={},
                    summary=f"실패: verdict={verdict_raw!r} 유효하지 않음",
                    success=False,
                    error=f"verdict는 {sorted(_VALID_VERDICTS)} 중 하나여야 합니다.",
                )
        verdict = VerdictType(verdict_raw)

        try:
            confidence = float(input_data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        explanation = (input_data.get("explanation") or "").strip()
        if not explanation:
            return ToolResult(
                output={},
                summary="실패: explanation 비어있음",
                success=False,
                error="explanation은 비울 수 없습니다.",
            )

        # ── 가드 (1-2 A안): evidence 한 번도 fetch 못 했는데 match/mismatch면 강등 ──
        # LLM이 fetch_evidence success 없이 finish(match, conf=1.0)을 호출하는
        # hallucination 차단. fetch가 한 번도 success로 끝난 적 없다면 어떤
        # 결론도 안전하지 않으므로 unverifiable로 강제 변환.
        if verdict in (VerdictType.MATCH, VerdictType.MISMATCH):
            if not _has_successful_fetch_evidence(context.workspace, context.claim_id):
                logger.warning(
                    f"[finish] {context.claim_id}: LLM verdict={verdict.value} "
                    f"호출했으나 fetch_evidence success 이력 0건 → unverifiable로 강등"
                )
                verdict = VerdictType.UNVERIFIABLE
                confidence = min(confidence, 0.3)
                explanation = (
                    "[자동 강등] LLM이 검증 완료로 보고했으나, 이 claim에 대해 "
                    "외부 데이터(fetch_evidence)를 한 번도 성공적으로 조회하지 못했습니다. "
                    "근거 없는 판정이므로 검증 불가로 처리합니다.\n\n"
                    f"원래 LLM 설명: {explanation[:300]}"
                )

        # data_points 파싱 (옵션)
        data_points_raw = input_data.get("data_points") or []
        data_points: list[DataPointSpec] = []
        for dp in data_points_raw:
            if isinstance(dp, dict):
                try:
                    data_points.append(DataPointSpec(**dp))
                except Exception as e:
                    logger.debug(f"[finish] data_point 파싱 실패: {dp} | {e}")

        # AgentVerdict 생성
        agent_verdict = AgentVerdict(
            claim_id=str(context.claim_id),
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            data_points=data_points,
            iterations_used=context.iter_num,
            stop_reason=StopReason.COMPLETED,
        )

        # workspace에 저장
        try:
            context.workspace.write_verdict(
                context.claim_id,
                agent_verdict.model_dump(mode="json"),
            )
        except Exception as e:
            logger.warning(f"[finish] verdict 저장 실패: {e}")

        # memory에 final 섹션 추가
        try:
            append_final(
                context.workspace,
                context.claim_id,
                verdict=verdict.value,
                confidence=confidence,
                reason=explanation[:200],  # 너무 길면 잘라서 memory에는 요약만
                iterations_used=context.iter_num,
            )
        except Exception as e:
            logger.debug(f"[finish] memory append 실패: {e}")

        return ToolResult(
            output={
                "verdict": verdict.value,
                "confidence": confidence,
                "iterations_used": context.iter_num,
                "data_points_count": len(data_points),
                "_finish": True,  # Loop에게 *종료 신호*
            },
            summary=(
                f"FINISH: verdict={verdict.value} confidence={confidence:.2f} "
                f"data_points={len(data_points)}"
            ),
            success=True,
        )
