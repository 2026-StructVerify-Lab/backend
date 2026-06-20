"""[리팩] agent Observation / verifier Evidence → decide_verdict 공통 입력 (판정 규칙 없음)"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from structverify.agent.schemas import ActionType, ClaimType, Plan
from structverify.core.schemas import (
    Claim,
    Evidence,
    MismatchType,
    VerificationResult,
    VerdictType,
)
from structverify.utils.logger import get_logger

from .row_match import (
    aggregate_rows_from_fetches,
    extract_criteria_from_row,
    find_value_for_time_with_criteria,
)

if TYPE_CHECKING:
    from structverify.graph.claim_graph import ClaimGraph
    from structverify.memory.working_memory import DocumentWorkingMemory

logger = get_logger(__name__)

_GROWTH_INDICATOR_KEYWORDS = ("증가율", "증감률", "증감율", "성장률", "비율", "퍼센트", "%")
_DIFF_INDICATOR_KEYWORDS = ("차이", "증감", "감소", "증가분", "감소분", "변화량", "격차")
_RANK_INDICATOR_KEYWORDS = ("순위", "1위", "최고", "최대", "최저", "최소", "가장 높")
_GROWTH_UNITS = ("%", "%p", "퍼센트", "%P", "pp")
_DERIVED_SUFFIXES = ("증가율", "감소율", "증감률", "변화율", "상승률", "하락률")


@dataclass
class VerdictDecision:
    """agent 경로 판정 결과 — loop가 AgentVerdict로 포장 (commit 8)."""

    claim_id: str
    verdict: VerdictType
    confidence: float
    explanation: str


@dataclass
class NormalizedInput:
    """decide_verdict가 소비하는 fallback 경로 입력."""

    evidence: Evidence
    claim_year: str | None = None
    claim_year_month: str | None = None


@dataclass
class AgentFetchInput:
    """agent fetch observation 판정 입력."""

    claim_id: str
    evidence: dict
    claim_actual_type: ClaimType
    plan_claim_type: ClaimType
    tolerance: float
    all_fetch_observations: list = field(default_factory=list)


@dataclass
class AgentCalculateInput:
    """agent calculate observation 판정 입력."""

    claim_id: str
    calc_value: float
    claim_actual_type: ClaimType
    calc_summary: str = ""


def infer_claim_type(claim: Claim) -> ClaimType | None:
    """loop._infer_claim_type — Planner LLM 분류 보정."""
    schema = claim.schema
    if schema is None:
        return None

    comp = getattr(schema, "comparison_type", None)
    if isinstance(comp, ClaimType):
        return comp

    canon = getattr(claim, "canonical_type", None)
    if isinstance(canon, ClaimType):
        return canon

    indicator = (schema.indicator or "").strip()
    unit = (schema.unit or "").strip()
    prev_value = getattr(schema, "prev_value", None)

    if any(kw in indicator for kw in _RANK_INDICATOR_KEYWORDS):
        return ClaimType.RANKING
    if any(kw in indicator for kw in _DIFF_INDICATOR_KEYWORDS):
        return ClaimType.DIFFERENCE
    if any(kw in indicator for kw in _GROWTH_INDICATOR_KEYWORDS):
        return ClaimType.GROWTH_RATE

    if prev_value is not None:
        if unit in _GROWTH_UNITS:
            return ClaimType.GROWTH_RATE
        return ClaimType.COMPARISON

    return ClaimType.ABSOLUTE


def from_agent_fetch(
    claim: Claim,
    last_observation: Any,
    plan: Plan,
    *,
    tolerance: float = 0.05,
    all_fetch_observations: list | None = None,
) -> tuple[AgentFetchInput | None, VerdictDecision | None]:
    """fetch Observation → AgentFetchInput 또는 즉시 VerdictDecision."""
    claim_id = str(claim.claim_id)

    if last_observation is None:
        return None, None
    if getattr(last_observation, "action", None) != ActionType.FETCH_EVIDENCE:
        return None, None

    if not getattr(last_observation, "success", False):
        summary = (getattr(last_observation, "summary", None) or "")[:200]
        return None, VerdictDecision(
            claim_id=claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.25,
            explanation=f"데이터 조회 실패: {summary}",
        )

    evidence = dict((getattr(last_observation, "output", None) or {}).get("evidence") or {})
    fetched_value = evidence.get("value")
    fetched_time = evidence.get("time_period", "") or ""

    schema = claim.schema
    claim_time = (schema.time_period or "") if schema is not None else ""

    matched_row_from_last = evidence.get("matched_row") or {}
    criteria = extract_criteria_from_row(matched_row_from_last)
    pool_rows = aggregate_rows_from_fetches(all_fetch_observations or [])
    for r in evidence.get("rows") or []:
        if isinstance(r, dict) and r not in pool_rows:
            pool_rows.append(r)

    if claim_time and pool_rows:
        hit = find_value_for_time_with_criteria(pool_rows, claim_time, criteria)
        if hit is not None:
            row_val_for_claim_time, _picked_row = hit
            claim_time_norm = str(claim_time).replace("-", "")
            fetched_time_norm = str(fetched_time).replace("-", "")
            if claim_time_norm not in fetched_time_norm:
                logger.info(
                    f"[loop] {claim_id}: aggregated rows에서 claim_time={claim_time} + "
                    f"criteria={list(criteria.keys()) or '없음'} row 매칭 "
                    f"→ value={row_val_for_claim_time} "
                    f"(마지막 fetch 시점={fetched_time}/value={fetched_value} → 덮어씀)"
                )
                evidence["value"] = row_val_for_claim_time
                evidence["time_period"] = claim_time_norm

    claim_actual_type = infer_claim_type(claim) or plan.claim_type

    return AgentFetchInput(
        claim_id=claim_id,
        evidence=evidence,
        claim_actual_type=claim_actual_type,
        plan_claim_type=plan.claim_type,
        tolerance=tolerance,
        all_fetch_observations=list(all_fetch_observations or []),
    ), None


def from_agent_calculate(
    claim: Claim,
    last_calc_observation: Any,
    plan: Plan,
    *,
    last_fetch_observation: Any | None = None,
    workspace: Any | None = None,
) -> tuple[AgentCalculateInput | None, VerdictDecision | None]:
    """calculate Observation → AgentCalculateInput (가드 통과 시)."""
    claim_id = str(claim.claim_id)

    if last_calc_observation is None or not getattr(last_calc_observation, "success", False):
        return None, None
    if getattr(last_calc_observation, "action", None) != ActionType.CALCULATE:
        return None, None

    if last_fetch_observation is None:
        sib_current: float | None = None
        try:
            sent_id = str(getattr(claim, "sent_id", "") or "").strip()
            if workspace is not None and sent_id and hasattr(workspace, "read_sibling_evidence"):
                sibs = workspace.read_sibling_evidence(sent_id) or []
                schema = claim.schema
                tp = (schema.time_period or "") if schema else ""
                tp_norm = str(tp).replace("-", "")
                for s in sibs:
                    if s.get("role") != "base":
                        continue
                    s_tp = str(s.get("time_period") or "").replace("-", "")
                    if s_tp == tp_norm and s.get("value") is not None:
                        sib_current = float(s.get("value"))
                        break
        except Exception:
            sib_current = None

        if sib_current is None:
            logger.info(
                f"[loop] {claim_id}: calculate 합성 가드 — fetch evidence 0건 + "
                f"sibling base도 없음 → calculate 결과 신뢰 X (LLM 환각 차단)"
            )
            return None, None

        calc_input = getattr(last_calc_observation, "input", None) or {}
        calc_current = calc_input.get("current")
        try:
            cc = float(calc_current) if calc_current is not None else None
        except (TypeError, ValueError):
            cc = None
        if cc is not None:
            gap_ratio = abs(cc - sib_current) / max(abs(sib_current), 1e-9)
            if gap_ratio > 0.02:
                logger.warning(
                    f"[loop] {claim_id}: calculate 합성 거부 — calc.input.current="
                    f"{cc} vs sibling base={sib_current} (gap {gap_ratio*100:.1f}%)"
                )
                return None, None

    schema = claim.schema
    schema_indicator = (schema.indicator or "").strip() if schema else ""
    if not any(schema_indicator.endswith(s) for s in _DERIVED_SUFFIXES):
        logger.info(
            f"[loop] {claim_id}: calculate 합성 가드 — base indicator "
            f"'{schema_indicator}' (derived 아님) → 합성 거부"
        )
        return None, None

    raw_result = (getattr(last_calc_observation, "output", None) or {}).get("result")
    if raw_result is None:
        return None, None
    try:
        calc_value = float(raw_result)
    except (TypeError, ValueError):
        return None, None

    if claim.schema is None or claim.schema.value is None:
        return None, None

    claim_actual_type = infer_claim_type(claim) or plan.claim_type
    calc_summary = getattr(last_calc_observation, "summary", None) or ""

    return AgentCalculateInput(
        claim_id=claim_id,
        calc_value=calc_value,
        claim_actual_type=claim_actual_type,
        calc_summary=calc_summary,
    ), None


def from_evidence(
    claim: Claim,
    evidence: Evidence | None,
    *,
    graph: ClaimGraph | None = None,
    memory: DocumentWorkingMemory | None = None,
) -> tuple[NormalizedInput | None, VerificationResult | None]:
    """Evidence·claim·graph·memory → NormalizedInput 또는 즉시 반환할 VerificationResult."""
    if evidence is None or evidence.official_value is None:
        return None, VerificationResult(
            claim_id=claim.claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            evidence=evidence,
        )

    claimed = claim.schema.value if claim.schema else None
    if claimed is None:
        return None, VerificationResult(
            claim_id=claim.claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2,
            evidence=None,
        )

    if claimed == 0.0:
        return None, VerificationResult(
            claim_id=claim.claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2,
            evidence=None,
        )

    if memory is not None and evidence.category_path:
        if not memory.domain_matches_category(evidence.category_path):
            logger.info(
                f"[verifier 도메인 가드] reject: "
                f"doc.domain={memory.domain} ↔ evidence.category={evidence.category_path}"
            )
            memory.record_stat_id_rejected(
                evidence.stat_table_id or "unknown",
                f"domain mismatch: {memory.domain} vs {evidence.category_path}",
            )
            return None, VerificationResult(
                claim_id=claim.claim_id,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.4,
                evidence=evidence,
                mismatch_type=MismatchType.DOMAIN_MISMATCH,
            )

    claim_year, claim_year_month = _resolve_claim_time(claim, graph)

    return NormalizedInput(
        evidence=evidence,
        claim_year=claim_year,
        claim_year_month=claim_year_month,
    ), None


def _resolve_claim_time(
    claim: Claim,
    graph: ClaimGraph | None,
) -> tuple[str | None, str | None]:
    claim_year = None
    claim_year_month = None

    schema_tp = (
        claim.schema.time_period if claim.schema and claim.schema.time_period else ""
    )

    if schema_tp:
        m = re.search(r"(\d{4})", schema_tp)
        if m:
            claim_year = m.group(1)
        ym = re.search(r"(\d{4})[-/]?(\d{2})", schema_tp)
        if ym:
            claim_year_month = ym.group(1) + ym.group(2)
        if claim_year:
            logger.info(
                f"[verifier] 시점 해소: schema.time_period={schema_tp!r} "
                f"→ year={claim_year}, ym={claim_year_month}"
            )

    if not claim_year and graph is not None:
        resolved = graph.resolve_time_for_claim(claim)
        if resolved:
            m = re.search(r"(\d{4})", resolved)
            if m:
                claim_year = m.group(1)
                logger.info(
                    f"[verifier] 시점 해소 (fallback): 그래프에서 resolved year={claim_year} "
                    f"(from {resolved})"
                )
            ym = re.search(r"(\d{4})[-/]?(\d{2})", resolved)
            if ym:
                claim_year_month = ym.group(1) + ym.group(2)

    return claim_year, claim_year_month
