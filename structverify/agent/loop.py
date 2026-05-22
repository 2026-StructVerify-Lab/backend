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
        _sent_id = ""
        _role = ""
        if claim is not None:
            _sent_id = str(getattr(claim, "sent_id", "") or "").strip()
            _schema = getattr(claim, "schema", None)
            _role = (getattr(_schema, "value_role", None) or "") if _schema else ""

        for dp in dps:
            val = getattr(dp, "resolved_value", None)
            if val is None:
                continue
            _fact = {
                "indicator": getattr(dp, "indicator", "") or "",
                "time_period": (getattr(dp, "source_time", None)
                                or getattr(dp, "time", "") or ""),
                # [2026-05-21] population 추가 — sub-claim별 격리 위해 캐시 키에 포함
                "population": (getattr(dp, "population", None) or ""),
                "value": val,
                "unit": getattr(dp, "resolved_unit", None) or "",
                "source": getattr(dp, "source", None) or "KOSIS",
                "claim_id": str(claim_id),
                "verdict": v_type,
            }
            workspace.append_verified_fact(_fact)
            # [S] sent_id 매핑에도 저장 — base 결과를 sibling derived가 활용
            if _sent_id and _role:
                workspace.record_sibling_evidence(
                    sent_id=_sent_id, role=_role, evidence=_fact,
                )
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


# ── [패치 H-3] 지표/시점 동시 매칭 row 찾기 helper ─────────────────────
# matched_row의 ITM_NM·C1_NM~C4_NM 컬럼을 criteria로 추출해, aggregated rows
# 풀에서 같은 지표(criteria)에 다른 시점(target_time)의 row를 찾는다.
# 여러 fetch observation의 rows[]를 합친 풀에서 시점만 가지고 row를 잡으면
# 다른 지표 row(예: 출생아 수와 혼인 건수가 같은 PRD_DE 공유)를 잘못 잡아
# 가짜 prev/current 비교를 만든다 — 그 버그를 차단.

_INDICATOR_CRITERIA_FIELDS = ("ITM_NM", "C1_NM", "C2_NM", "C3_NM", "C4_NM")


def _extract_criteria_from_row(row: dict) -> dict:
    """matched_row에서 지표 식별 컬럼만 추출."""
    if not isinstance(row, dict):
        return {}
    return {
        k: row[k]
        for k in _INDICATOR_CRITERIA_FIELDS
        if k in row and row[k] is not None and str(row[k]).strip() != ""
    }


def _find_value_for_time_with_criteria(
    all_rows: list[dict],
    target_time: str,
    criteria: dict | None,
) -> tuple[float, dict] | None:
    """rows[]에서 target_time 매칭 + criteria 컬럼 값 일치하는 row 찾기.

    criteria가 비면 _find_row_value_for_time과 동일 동작.
    찾으면 (DT 값, 매칭한 row) 반환.
    """
    if not all_rows or not target_time:
        return None
    norm = str(target_time).replace("-", "").strip()

    def _row_matches_criteria(row: dict) -> bool:
        if not criteria:
            return True
        for k, v in criteria.items():
            if str(row.get(k, "")).strip() != str(v).strip():
                return False
        return True

    # 1차: PRD_DE 완전 일치 + criteria 일치
    for row in all_rows:
        if not isinstance(row, dict):
            continue
        prd = str(row.get("PRD_DE", "") or "").strip()
        if prd != norm:
            continue
        if not _row_matches_criteria(row):
            continue
        v = _parse_row_dt(row.get("DT"))
        if v is not None:
            return (v, row)

    # 2차: 연 단위 fallback (PRD_DE='YYYY')
    year = norm[:4]
    if year and year != norm:
        for row in all_rows:
            if not isinstance(row, dict):
                continue
            prd = str(row.get("PRD_DE", "") or "").strip()
            if prd != year:
                continue
            if not _row_matches_criteria(row):
                continue
            v = _parse_row_dt(row.get("DT"))
            if v is not None:
                return (v, row)
    return None


def _aggregate_rows_from_fetches(
    fetch_observations: list,
) -> list[dict]:
    """여러 fetch observation의 rows[]를 평탄화해 합친 리스트."""
    out: list[dict] = []
    if not fetch_observations:
        return out
    seen_ids: set[int] = set()
    for obs in fetch_observations:
        ev = (getattr(obs, "output", None) or {}).get("evidence") or {}
        rs = ev.get("rows") or []
        for r in rs:
            if not isinstance(r, dict):
                continue
            rid = id(r)
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            out.append(r)
    return out


def _try_growth_rate_from_rows(
    evidence: dict,
    schema: Any,
    claim_id: str,
    all_fetch_observations: list | None = None,
) -> tuple[float, float, float, str] | None:
    """[v6.17] growth_rate claim을 같은 표 rows로 직접 계산.

    KOSIS에 '증가율' 통계표가 따로 없을 때, fetch한 표(현재값 표)의 rows에서
    prev_time_period 시점의 행을 찾아 (current-prev)/prev*100 을 직접 계산한다.
    추가 API 호출 불필요 — evidence dict에 rows 전체가 들어있음.

    [패치 H-3] all_fetch_observations가 주어지면 pool로 사용. evidence.value를
    무조건 current로 쓰지 않고 claim_time(현재 시점) row를 풀에서 다시 찾음.
    이렇게 해야 last fetch가 prev_time이었던 경우에 current_val이 prev's value로
    엉뚱하게 잡히는 버그를 막는다. 또 prev row도 matched_row의 지표 criteria로
    필터해 다른 지표 row(예: 출생아 수 vs 혼인 건수)가 잘못 매칭되는 걸 차단.

    반환: (계산된_증가율, current_value, prev_value, 설명) 또는 None.
    """
    prev_time = getattr(schema, "prev_time_period", None) if schema else None
    if not prev_time:
        return None  # 비교 시점 없음 → 계산 불가

    cur_time = getattr(schema, "time_period", None) if schema else None

    # rows pool: last fetch + 모든 fetch observation rows 합집합
    rows = list(evidence.get("rows") or [])
    pool_rows: list[dict] = []
    if all_fetch_observations:
        pool_rows = _aggregate_rows_from_fetches(all_fetch_observations)
    for r in rows:
        if isinstance(r, dict) and r not in pool_rows:
            pool_rows.append(r)
    if not pool_rows:
        return None

    # matched_row criteria로 같은 지표 row만 후보 추리기
    matched_row = evidence.get("matched_row") or {}
    criteria = _extract_criteria_from_row(matched_row)

    # current_val: claim_time(현재) row를 풀에서 다시 찾음. 못 찾으면 evidence.value 사용
    current_val: float | None = None
    if cur_time:
        cur_hit = _find_value_for_time_with_criteria(pool_rows, cur_time, criteria)
        if cur_hit is not None:
            current_val, _ = cur_hit
    if current_val is None:
        current_val = _parse_row_dt(evidence.get("value"))
        if current_val is None and matched_row:
            current_val = _parse_row_dt(matched_row.get("DT"))
    if current_val is None:
        return None

    # prev_val: 같은 지표 criteria로 prev_time 시점 row 탐색
    prev_hit = _find_value_for_time_with_criteria(pool_rows, prev_time, criteria)
    if prev_hit is None:
        # criteria 너무 좁아서 못 찾았으면 criteria 없이 한 번 더 (안전망)
        # 다만 이건 마지막 수단 — log로 명시
        logger.info(
            f"[loop] {claim_id}: growth_rate 직접계산 — criteria 매칭 prev row "
            f"{prev_time!r} 못 찾음. 지표 무관 시점 매칭으로 fallback "
            f"(pool={len(pool_rows)} rows, criteria={list(criteria.keys()) or '없음'})"
        )
        prev_val_legacy = _find_row_value_for_time(pool_rows, prev_time)
        if prev_val_legacy is None:
            return None
        prev_val = prev_val_legacy
    else:
        prev_val, _ = prev_hit

    if prev_val == 0:
        return None  # 0으로 나눗셈 방지

    calc_rate = (current_val - prev_val) / prev_val * 100.0
    desc = (
        f"표에서 직접 계산: 현재값({cur_time or '?'}) {current_val} - "
        f"이전값({prev_time}) {prev_val} "
        f"→ 증가율 ({current_val}-{prev_val})/{prev_val}×100 = {calc_rate:.2f}%"
    )
    logger.info(f"[loop] {claim_id}: growth_rate 직접계산 성공 — {desc}")
    return (calc_rate, current_val, prev_val, desc)


def _try_difference_from_rows(
    evidence: dict,
    schema: Any,
    claim_id: str,
    all_fetch_observations: list | None = None,
) -> tuple[float, float, float, str] | None:
    """[수정 v6.23] difference claim을 같은 표 rows로 직접 계산.

    growth_rate와 동일한 원리 — fetch한 표의 rows에서 prev_time_period
    시점 행을 찾아 (current - prev) 차이를 직접 계산한다.

    [패치 H-3] aggregated pool + criteria 필터로 current/prev row를 올바르게
    잡는다. evidence.value 무조건 사용 안 함.

    반환: (계산된_차이, current_value, prev_value, 설명) 또는 None.
    """
    prev_time = getattr(schema, "prev_time_period", None) if schema else None
    if not prev_time:
        return None  # 비교 시점 없음 → 계산 불가

    cur_time = getattr(schema, "time_period", None) if schema else None

    rows = list(evidence.get("rows") or [])
    pool_rows: list[dict] = []
    if all_fetch_observations:
        pool_rows = _aggregate_rows_from_fetches(all_fetch_observations)
    for r in rows:
        if isinstance(r, dict) and r not in pool_rows:
            pool_rows.append(r)
    if not pool_rows:
        return None

    matched_row = evidence.get("matched_row") or {}
    criteria = _extract_criteria_from_row(matched_row)

    current_val: float | None = None
    if cur_time:
        cur_hit = _find_value_for_time_with_criteria(pool_rows, cur_time, criteria)
        if cur_hit is not None:
            current_val, _ = cur_hit
    if current_val is None:
        current_val = _parse_row_dt(evidence.get("value"))
        if current_val is None and matched_row:
            current_val = _parse_row_dt(matched_row.get("DT"))
    if current_val is None:
        return None

    prev_hit = _find_value_for_time_with_criteria(pool_rows, prev_time, criteria)
    if prev_hit is None:
        logger.info(
            f"[loop] {claim_id}: difference 직접계산 — criteria 매칭 prev row "
            f"{prev_time!r} 못 찾음. 지표 무관 fallback "
            f"(pool={len(pool_rows)} rows, criteria={list(criteria.keys()) or '없음'})"
        )
        prev_val_legacy = _find_row_value_for_time(pool_rows, prev_time)
        if prev_val_legacy is None:
            return None
        prev_val = prev_val_legacy
    else:
        prev_val, _ = prev_hit

    calc_diff = current_val - prev_val
    desc = (
        f"표에서 직접 계산: 현재값({cur_time or '?'}) {current_val} - "
        f"이전값({prev_time}) {prev_val} → 차이 {current_val}-{prev_val} = {calc_diff:.4f}"
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
    all_fetch_observations: list | None = None,
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

    # ── [패치 H-3] aggregated rows에서 claim_time + 지표 criteria 매칭 row 찾기 ──
    # 시나리오: LLM이 current(2025-04) fetch → prev(2024-04) fetch 순으로 호출하면
    # last_fetch_observation은 prev 시점만 들어있고 그 fetch의 rows[]에는 2025-04
    # row가 아예 없다. 단일 fetch만 보면 claim_time row 못 찾아 unverifiable.
    # → 같은 claim의 모든 fetch observation rows를 합쳐서 풀을 만들고,
    #   matched_row의 ITM_NM·C1_NM~C4_NM을 criteria로 같은 지표의 다른 시점 row를
    #   찾는다. 시점만 보고 row 잡으면 출생아 수/혼인 건수 같이 PRD_DE 공유하는
    #   다른 지표가 잘못 매칭됨 — criteria 필터로 차단.
    matched_row_from_last = evidence.get("matched_row") or {}
    criteria = _extract_criteria_from_row(matched_row_from_last)
    pool_rows = _aggregate_rows_from_fetches(all_fetch_observations or [])
    # last fetch의 rows도 합집합에 포함 (보통은 이미 포함됐을 것이나 안전)
    last_evidence_rows = evidence.get("rows") or []
    for r in last_evidence_rows:
        if isinstance(r, dict) and r not in pool_rows:
            pool_rows.append(r)

    if claim_time and pool_rows:
        hit = _find_value_for_time_with_criteria(pool_rows, claim_time, criteria)
        if hit is not None:
            row_val_for_claim_time, _picked_row = hit
            claim_time_norm = str(claim_time).replace("-", "")
            fetched_time_norm = str(fetched_time).replace("-", "")
            # 마지막 fetch가 이미 claim_time이면 그대로, 아니면 덮어씀
            if claim_time_norm not in fetched_time_norm:
                logger.info(
                    f"[loop] {claim_id}: aggregated rows에서 claim_time={claim_time} + "
                    f"criteria={list(criteria.keys()) or '없음'} row 매칭 "
                    f"→ value={row_val_for_claim_time} "
                    f"(마지막 fetch 시점={fetched_time}/value={fetched_value} → 덮어씀)"
                )
                fetched_value = row_val_for_claim_time
                fetched_time = claim_time_norm

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
            calc = _try_growth_rate_from_rows(
                evidence, schema, claim_id,
                all_fetch_observations=all_fetch_observations,
            )
            if calc is not None and claim_value is not None:
                calc_rate, cur_v, prev_v, calc_desc = calc
                try:
                    claimed_rate = float(claim_value)
                except (TypeError, ValueError):
                    claimed_rate = None
                if claimed_rate is not None:
                    # ── [패치 J] 증가율/감소율 부호 방향 가드 ─────────────
                    # 기사 "혼인 건수 증가율 4.9%"는 양의 방향(증가) 주장.
                    # 시스템 계산 -5.25% (감소)는 정반대 방향이지만, 기존엔
                    # abs(abs(-5.25)-abs(4.9))=0.35 ≤ 1.5 로 MATCH 통과시켰음.
                    # 가짜 일치를 만들어 데이터 신뢰성을 깨므로, indicator의
                    # 방향 단서(증가율/상승률 vs 감소율/하락률)와 calc_rate
                    # 부호가 어긋나면 즉시 MISMATCH로 떨어트린다.
                    _INCREASE_SFX = ("증가율", "상승률")
                    _DECREASE_SFX = ("감소율", "하락률")
                    _ind = (claim_indicator or "").strip()
                    _expects_inc = any(_ind.endswith(s) for s in _INCREASE_SFX)
                    _expects_dec = any(_ind.endswith(s) for s in _DECREASE_SFX)
                    _direction_mismatch = (
                        (_expects_inc and calc_rate < 0)
                        or (_expects_dec and calc_rate > 0)
                    )
                    if _direction_mismatch:
                        diff = abs(abs(calc_rate) - abs(claimed_rate))
                        logger.warning(
                            f"[loop] {claim_id}: growth_rate 부호 방향 불일치 "
                            f"(indicator={_ind!r}, 기사 {claimed_rate:+.2f}% 방향, "
                            f"계산 {calc_rate:+.2f}% 반대 방향) → MISMATCH 강제"
                        )
                        return AgentVerdict(
                            claim_id=claim_id,
                            verdict=VerdictType.MISMATCH,
                            confidence=0.75,
                            explanation=(
                                f"증가율 방향 불일치: 기사는 '{_ind}' "
                                f"{claimed_rate}% (양의 방향), "
                                f"KOSIS({stat_table_id}) 표 계산값 "
                                f"{calc_rate:.2f}% ({'감소' if calc_rate < 0 else '증가'} 방향). "
                                f"{calc_desc}"
                            ),
                            data_points=_evidence_to_data_points(evidence, claim),
                            iterations_used=iter_num,
                            stop_reason=StopReason.COMPLETED,
                        )
                    # 부호 일치 — 부호 무시하고 절대 크기 비교
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
            calc = _try_difference_from_rows(
                evidence, schema, claim_id,
                all_fetch_observations=all_fetch_observations,
            )
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


def _synthesize_verdict_from_calculate(
    plan: Plan,
    claim: Any,
    claim_id: str,
    last_calc_observation: Observation | None,
    iter_num: int,
    last_fetch_observation: Observation | None = None,
    workspace: Any = None,
) -> AgentVerdict | None:
    """[패치] Plan 소진 + 마지막 성공한 관측이 CALCULATE인 경우 verdict 합성.

    LLM이 prev/current를 계산했지만 finish를 안 부르고 다시 같은 액션 반복
    → 중복차단 → 강제 unverifiable로 죽는 케이스(2026-05-20 진단). calculate
    output의 result 값을 claim.schema.value와 비교해 자동 verdict 생성한다.

    growth_rate/difference claim에서 LLM이 계산 결과를 8.6993로 얻었는데
    기사 주장 8.7%와 일치해도 finish를 못 부르면 그동안 unverifiable로 끝났음.

    [안전장치] last_fetch_observation이 없으면 calculate 결과를 신뢰하지
    않는다. fetch 0건 상태에서 LLM이 prev/current를 임의로 박아 계산한
    값(예: prev=18123 같이 출처 불명 값)이 우연히 article과 비슷하다고
    MATCH로 만들어 환각을 통과시키는 걸 차단.

    [P22 2026-05-22] sibling base evidence가 있으면 fetch 0건이어도 합성 시도.
    derived claim이 같은 문장 base sub-claim의 KOSIS 값을 cache로 활용하는
    경로(혼인 건수 base 18919 → 증가율 claim이 fetch 없이 calc) 회복. 단,
    calc.input.current가 sibling base value와 *크게 다르면(>2%)* 환각으로 거부.
    """
    if last_calc_observation is None or not last_calc_observation.success:
        return None
    if last_calc_observation.action != ActionType.CALCULATE:
        return None
    if last_fetch_observation is None:
        # [P22] sibling 검증으로 fetch 0건 케이스 구제 시도
        _sib_current: float | None = None
        try:
            _sent_id = str(getattr(claim, "sent_id", "") or "").strip()
            if workspace is not None and _sent_id and hasattr(workspace, "read_sibling_evidence"):
                _sibs = workspace.read_sibling_evidence(_sent_id) or []
                # 같은 시점의 base sibling 찾기
                _schema = getattr(claim, "schema", None)
                _tp = (getattr(_schema, "time_period", None) or "") if _schema else ""
                _tp_norm = str(_tp).replace("-", "")
                for _s in _sibs:
                    if _s.get("role") != "base":
                        continue
                    _s_tp = str(_s.get("time_period") or "").replace("-", "")
                    if _s_tp == _tp_norm and _s.get("value") is not None:
                        _sib_current = float(_s.get("value"))
                        break
        except Exception:
            _sib_current = None

        if _sib_current is None:
            logger.info(
                f"[loop] {claim_id}: calculate 합성 가드 — fetch evidence 0건 + "
                f"sibling base도 없음 → calculate 결과 신뢰 X (LLM 환각 차단)"
            )
            return None

        # calc input의 current와 sibling base value 비교
        _calc_input = last_calc_observation.input or {}
        _calc_current = _calc_input.get("current")
        try:
            _cc = float(_calc_current) if _calc_current is not None else None
        except (TypeError, ValueError):
            _cc = None
        if _cc is not None:
            _gap_ratio = abs(_cc - _sib_current) / max(abs(_sib_current), 1e-9)
            if _gap_ratio > 0.02:
                logger.warning(
                    f"[loop] {claim_id}: calculate 합성 거부 — calc.input.current="
                    f"{_cc} vs sibling base={_sib_current} (gap {_gap_ratio*100:.1f}%) "
                    f"→ LLM이 sibling 무시하고 환각 가능성 (혼인 건수 4.9% 케이스 회귀)"
                )
                return None
            logger.info(
                f"[loop] {claim_id}: calculate 합성 — fetch 0건이지만 sibling base"
                f"({_sib_current}) ≈ calc.current({_cc}) → calc 결과 신뢰"
            )
        # else: calc input에 current 없음 — sibling 있으면 그래도 시도

    # [패치 2026-05-20] base claim은 calculate 합성 거부.
    # planner가 base claim(예: '출생아 수 20717명')에 plan.type=GROWTH_RATE로
    # 잘못 분류한 경우, calculate 결과(8.825%)와 claim value(20717명)를 강제
    # 비교해 오차 99.96% MISMATCH로 잘못 떨어지는 걸 차단. derived suffix가
    # 명시된 claim(~증가율, ~감소율 등)에만 calculate 합성 적용.
    _schema = getattr(claim, "schema", None)
    _schema_indicator = (
        (getattr(_schema, "indicator", "") or "").strip() if _schema else ""
    )
    _DERIVED_SUFFIXES = (
        "증가율", "감소율", "증감률", "변화율", "상승률", "하락률",
    )
    if not any(_schema_indicator.endswith(s) for s in _DERIVED_SUFFIXES):
        logger.info(
            f"[loop] {claim_id}: calculate 합성 가드 — base indicator "
            f"'{_schema_indicator}' (derived 아님) → calculate 결과는 "
            f"단위가 다르므로 합성 거부"
        )
        return None

    raw_result = (last_calc_observation.output or {}).get("result")
    if raw_result is None:
        return None
    try:
        calc_value = float(raw_result)
    except (TypeError, ValueError):
        return None

    schema = getattr(claim, "schema", None)
    claim_value = getattr(schema, "value", None) if schema is not None else None
    claim_unit = (getattr(schema, "unit", "") or "") if schema is not None else ""
    if claim_value is None:
        return None
    try:
        cv = float(claim_value)
    except (TypeError, ValueError):
        return None

    claim_actual_type = _infer_claim_type(claim) or plan.claim_type

    # claim type별 비교 — growth_rate/difference는 부호 무시 절댓값 비교
    if isinstance(claim_actual_type, ClaimType) and claim_actual_type == ClaimType.GROWTH_RATE:
        diff = abs(abs(calc_value) - abs(cv))
        if diff <= 1.5:
            verdict_t, conf, label = VerdictType.MATCH, 0.8, "일치"
        elif diff <= 5.0:
            verdict_t, conf, label = VerdictType.UNVERIFIABLE, 0.4, "오차 큼"
        else:
            verdict_t, conf, label = VerdictType.MISMATCH, 0.7, "불일치"
        diff_desc = f"차이 {diff:.2f}%p"
    elif isinstance(claim_actual_type, ClaimType) and claim_actual_type == ClaimType.DIFFERENCE:
        gap = abs(abs(calc_value) - abs(cv))
        tol = max(abs(cv) * 0.10, 0.02)
        if gap <= tol:
            verdict_t, conf, label = VerdictType.MATCH, 0.8, "일치"
        elif gap <= tol * 3:
            verdict_t, conf, label = VerdictType.UNVERIFIABLE, 0.4, "오차 큼"
        else:
            verdict_t, conf, label = VerdictType.MISMATCH, 0.7, "불일치"
        diff_desc = f"차이 {gap:.4f}, 허용 {tol:.4f}"
    else:
        # 일반 비교 — 1% 오차
        if abs(cv) < 1e-9:
            diff_ratio = 0.0 if abs(calc_value) < 1e-9 else 1.0
        else:
            diff_ratio = abs(calc_value - cv) / abs(cv)
        if diff_ratio < 0.01:
            verdict_t, conf, label = VerdictType.MATCH, 0.8, "일치"
        else:
            verdict_t, conf, label = VerdictType.MISMATCH, 0.7, "불일치"
        diff_desc = f"오차 {diff_ratio*100:.2f}%"

    logger.info(
        f"[loop] {claim_id}: calculate 합성 판정={label} "
        f"(기사 {cv}{claim_unit} vs 계산 {calc_value:.4g}, {diff_desc})"
    )
    return AgentVerdict(
        claim_id=claim_id,
        verdict=verdict_t,
        confidence=conf,
        explanation=(
            f"Agent가 직접 계산한 결과로 검증: 기사 주장 {cv}{claim_unit}, "
            f"산출된 값 {calc_value:.4g} ({diff_desc}). "
            f"계산식: {(last_calc_observation.summary or '')[:200]}"
        ),
        data_points=[],
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