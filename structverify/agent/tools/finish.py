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


def _collect_fetch_evidences(workspace, claim_id) -> list[dict]:
    """[2026-05-25] 이 claim의 모든 successful fetch_evidence observation을
    {indicator, time_period, value, unit, stat_id} dict로 평탄화해 모음.

    FinishTool이 LLM이 채운 data_points의 resolved_value를 *실제 fetched value*로
    덮어쓰기 위한 ground truth. LLM이 claim value를 그대로 박는 hallucination 방지.

    indicator 우선순위: evidence dict의 indicator → observation의 params.indicator.
    (EvidenceData는 indicator 필드가 없는 경우가 있어 fallback으로 params 사용)
    """
    out: list[dict] = []
    try:
        names = workspace.list_observations(claim_id)
    except Exception:
        return out
    for name in names:
        if "fetch" not in name.lower():
            continue
        data = workspace.read_observation(claim_id, name)
        if not isinstance(data, dict):
            continue
        ev = (data.get("output") or {}).get("evidence") or data.get("evidence") or {}
        if not isinstance(ev, dict) or ev.get("value") is None:
            continue
        # indicator는 evidence에 보통 비어있어 params에서 보강
        _ind = str(ev.get("indicator") or "").strip()
        if not _ind:
            _params = data.get("params") or (data.get("output") or {}).get("params") or {}
            if isinstance(_params, dict):
                _ind = str(_params.get("indicator") or "").strip()
        out.append({
            "indicator": _ind,
            "time_period": str(ev.get("time_period") or ""),
            "value": ev.get("value"),
            "unit": str(ev.get("unit") or ""),
            "stat_id": str(ev.get("stat_table_id") or ""),
            "obs_name": name,  # 디버깅용 trace
        })
    return out


def _match_evidence_for_data_point(dp_dict: dict, evidences: list[dict]) -> dict | None:
    """data_point의 (indicator, time)으로 가장 가까운 fetch evidence 찾기.

    매칭 룰:
      1) indicator + time 정규화 후 완전 일치
      2) time만 일치 (indicator는 LLM이 다르게 표기 가능)
      3) indicator만 일치
    매칭 안 되면 None — LLM이 채운 값 그대로 유지 (calculate 결과 등).
    """
    def _norm_time(t: str) -> str:
        if not t:
            return ""
        s = str(t).strip().replace("-", "").replace(".", "").replace("/", "")
        return s
    def _norm_ind(s: str) -> str:
        if not s:
            return ""
        return str(s).strip().replace(" ", "").lower()

    dp_ind = _norm_ind(dp_dict.get("indicator", ""))
    dp_time = _norm_time(dp_dict.get("time") or dp_dict.get("source_time") or "")

    # 1차: indicator + time 둘 다 일치
    for ev in evidences:
        if _norm_ind(ev["indicator"]) == dp_ind and _norm_time(ev["time_period"]).startswith(dp_time):
            return ev
        if _norm_ind(ev["indicator"]) == dp_ind and dp_time.startswith(_norm_time(ev["time_period"])):
            return ev
    # 2차: time만 일치
    if dp_time:
        for ev in evidences:
            ev_t = _norm_time(ev["time_period"])
            if ev_t == dp_time or ev_t.startswith(dp_time) or dp_time.startswith(ev_t):
                return ev
    # 3차: indicator만 일치 (time 없는 경우 등)
    if dp_ind:
        for ev in evidences:
            if _norm_ind(ev["indicator"]) == dp_ind:
                return ev
    return None


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
        # [2026-05-25] LLM이 채운 resolved_value가 claim 값을 그대로 박는 경우가 있음
        # (예: claim "20717명" → LLM이 evidence value도 20717이라고 보고 → 실제 KOSIS
        # 값은 20787인데 20717로 저장됨 → verified_facts 캐시도 오염).
        # 대응: workspace의 실제 fetch_evidence observation에서 indicator/time이
        # 매칭되는 값을 찾아 resolved_value를 *evidence의 값으로 덮음*.
        # 매칭 안 되는 data_point는 LLM 값 유지 (계산 결과 등).
        evidences = _collect_fetch_evidences(context.workspace, context.claim_id)

        data_points_raw = input_data.get("data_points") or []

        # 진단: evidence pool과 LLM이 채운 data_points를 한 번에 로그
        logger.info(
            f"[finish] {context.claim_id}: evidence pool ({len(evidences)}건) ↓\n"
            + "\n".join(
                f"  - ev[{i}] indicator={e['indicator']!r} time={e['time_period']!r} "
                f"value={e['value']!r} unit={e['unit']!r} (obs={e['obs_name']})"
                for i, e in enumerate(evidences)
            )
            + f"\n  LLM data_points_raw ({len(data_points_raw)}건): {data_points_raw}"
        )

        data_points: list[DataPointSpec] = []
        # 이미 data_point로 덮인 (indicator,time)을 추적 — 중복 추가 방지
        _covered_keys: set[tuple[str, str]] = set()

        def _norm_t(t: str) -> str:
            return str(t or "").strip().replace("-", "").replace(".", "")
        def _norm_i(s: str) -> str:
            return str(s or "").strip().replace(" ", "").lower()

        for dp in data_points_raw:
            if not isinstance(dp, dict):
                continue
            matched_ev = _match_evidence_for_data_point(dp, evidences)
            if matched_ev is None:
                logger.info(
                    f"[finish] {context.claim_id}: data_point 매칭 evidence 없음 — "
                    f"LLM 값 유지 (indicator={dp.get('indicator')!r}, "
                    f"time={dp.get('time')!r}, value={dp.get('resolved_value')!r}). "
                    f"이유: evidence pool에 해당 (indicator,time) 없음 또는 정규화 mismatch."
                )
            else:
                _llm_val = dp.get("resolved_value")
                _ev_val = matched_ev["value"]
                if _llm_val == _ev_val:
                    logger.info(
                        f"[finish] {context.claim_id}: data_point resolved_value 일치 "
                        f"(LLM=evidence={_ev_val}) — 보정 불필요 "
                        f"(indicator={dp.get('indicator')!r}, time={dp.get('time')!r})"
                    )
                else:
                    logger.info(
                        f"[finish] {context.claim_id}: data_point resolved_value 보정 — "
                        f"LLM={_llm_val} → evidence={_ev_val} "
                        f"(indicator={dp.get('indicator')!r}, time={dp.get('time')!r}, "
                        f"src_obs={matched_ev['obs_name']})"
                    )
                # evidence ground-truth로 덮어씀 (resolved_value + resolved_unit + source)
                dp = {
                    **dp,
                    "resolved_value": _ev_val,
                    "resolved_unit": matched_ev["unit"] or dp.get("resolved_unit"),
                    "source": (
                        f"KOSIS:{matched_ev['stat_id']}"
                        if matched_ev["stat_id"] else dp.get("source") or "KOSIS"
                    ),
                    "source_time": matched_ev["time_period"] or dp.get("source_time"),
                }
            try:
                _spec = DataPointSpec(**dp)
                data_points.append(_spec)
                _covered_keys.add((_norm_i(_spec.indicator), _norm_t(_spec.source_time or _spec.time)))
            except Exception as e:
                logger.debug(f"[finish] data_point 파싱 실패: {dp} | {e}")

        # [2026-05-25] 빠진 evidence 자동 보강 — LLM이 data_points에 안 박았어도
        # workspace에 fetch_evidence success가 있으면 시스템이 보장해서 data_points에
        # 추가. 이렇게 해야 verified_facts 캐시에 *모든 fetched 값*이 저장되어
        # 다음 claim(예: 증가율)이 prev/current 둘 다 재검색 없이 즉시 가져옴.
        # LLM이 1개만 박아도 시스템이 나머지를 보강하므로 derived claim 효율 보장.
        for ev in evidences:
            _ind = ev["indicator"]
            if not _ind or ev["value"] is None:
                continue
            _key = (_norm_i(_ind), _norm_t(ev["time_period"]))
            if _key in _covered_keys:
                continue
            try:
                _spec = DataPointSpec(
                    indicator=_ind,
                    time=ev["time_period"],
                    resolved_value=ev["value"],
                    resolved_unit=ev["unit"] or None,
                    source=(f"KOSIS:{ev['stat_id']}" if ev["stat_id"] else "KOSIS"),
                    source_time=ev["time_period"] or None,
                )
                data_points.append(_spec)
                _covered_keys.add(_key)
                logger.info(
                    f"[finish] {context.claim_id}: evidence 자동 보강 — "
                    f"indicator={_ind!r} time={ev['time_period']!r} "
                    f"value={ev['value']!r} (LLM이 data_points에 안 넣음, "
                    f"src_obs={ev['obs_name']}) — verified_facts 캐시 보존용."
                )
            except Exception as e:
                logger.debug(f"[finish] evidence 자동 보강 실패: {ev} | {e}")

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

        # [2026-05-25] 디버깅/UI 추적용 — claim 디렉토리에 최종 data_points 덤프.
        # 어떤 LLM raw vs evidence-corrected 값으로 결론이 났는지 추적 가능.
        try:
            dp_dump = {
                "claim_id": str(context.claim_id),
                "verdict": verdict.value,
                "confidence": confidence,
                "iter_num": context.iter_num,
                "evidences_collected": [
                    {
                        "indicator": e["indicator"], "time_period": e["time_period"],
                        "value": e["value"], "unit": e["unit"],
                        "stat_id": e["stat_id"], "obs": e["obs_name"],
                    } for e in evidences
                ],
                "llm_data_points_raw": data_points_raw,
                "final_data_points": [dp.model_dump(mode="json") for dp in data_points],
            }
            context.workspace.write_observation(
                context.claim_id, "_final_data_points", dp_dump,
            )
            context.workspace.write_data_points(
                context.claim_id, [dp.model_dump(mode="json") for dp in data_points],
            )
        except Exception as e:
            logger.debug(f"[finish] data_points dump 실패 (무시): {e}")

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
