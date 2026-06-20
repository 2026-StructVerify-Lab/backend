"""[리팩] Step 8 판정 메인 진입 — profile별 분기 (fallback 우선)"""
from __future__ import annotations

import re

from structverify.core.schemas import (
    Claim,
    Evidence,
    MismatchType,
    VerificationResult,
    VerdictType,
)
from structverify.utils.logger import get_logger

from ._config import VerificationProfile
from .adapters import (
    AgentCalculateInput,
    AgentFetchInput,
    NormalizedInput,
    VerdictDecision,
)
from .decide_verdict_agent import (
    decide_verdict_agent_calculate,
    decide_verdict_agent_fetch,
)
from .growth_diff import verify_growth_or_diff
from .row_match import extract_numeric_values, find_best_match
from .units import is_same_unit_type, normalize_value
from .verdict_thresholds import verdict_from_error

logger = get_logger(__name__)


def decide_verdict(
    claim: Claim,
    normalized: NormalizedInput | AgentFetchInput | AgentCalculateInput,
    config: dict | None = None,
    profile: VerificationProfile = "fallback",
) -> VerificationResult | VerdictDecision:
    """claim + 정규화된 입력 → VerificationResult(fallback) 또는 VerdictDecision(agent)."""
    config = config or {}
    if profile == "fallback":
        if not isinstance(normalized, NormalizedInput):
            raise TypeError("profile='fallback' requires NormalizedInput")
        return _decide_verdict_fallback(claim, normalized, config)
    if profile == "agent":
        if isinstance(normalized, AgentFetchInput):
            return decide_verdict_agent_fetch(claim, normalized, config)
        if isinstance(normalized, AgentCalculateInput):
            return decide_verdict_agent_calculate(claim, normalized, config)
        raise TypeError("profile='agent' requires AgentFetchInput or AgentCalculateInput")
    raise NotImplementedError(f"profile={profile!r}")


def _decide_verdict_fallback(
    claim: Claim,
    normalized: NormalizedInput,
    config: dict,
) -> VerificationResult:
    evidence = normalized.evidence
    claim_year = normalized.claim_year
    claim_year_month = normalized.claim_year_month

    claimed = claim.schema.value if claim.schema else None
    claim_unit = (claim.schema.unit or "") if claim.schema else ""

    prev_value = getattr(claim.schema, "prev_value", None) if claim.schema else None
    if prev_value is not None and prev_value != 0:
        indicator = (claim.schema.indicator or "") if claim.schema else ""
        is_ratio_schema = claim_unit and (
            "%" in claim_unit
            or "퍼센트" in claim_unit
            or "율" in claim_unit
            or "비율" in claim_unit
        )
        is_diff_schema = (
            "차이" in indicator or "증감" in indicator or "변화량" in indicator
        )
        if is_ratio_schema or is_diff_schema:
            calc_result = verify_growth_or_diff(
                claim,
                evidence,
                claim_year,
                claim_year_month,
                prev_value,
                is_ratio_schema,
                config,
                classify_mismatch=_classify_mismatch,
            )
            if calc_result is not None:
                return calc_result

    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    rows = raw.get("row", [])
    if isinstance(rows, list) and rows:
        kosis_values = extract_numeric_values(rows)
        if kosis_values:
            best_match, best_error = find_best_match(
                claimed,
                claim_unit,
                claim_year,
                kosis_values,
                claim_year_month=claim_year_month,
            )

            if best_match is None:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    verdict=VerdictType.UNVERIFIABLE,
                    confidence=0.3,
                    evidence=evidence,
                )

            evidence = evidence.model_copy(update={
                "official_value": best_match.get("value"),
                "unit": best_match.get("unit") or evidence.unit,
                "time_period": best_match.get("period") or evidence.time_period,
            })
            logger.info(
                f"[verifier] evidence 동기화 (F2): official_value={evidence.official_value} "
                f"unit={evidence.unit!r} time_period={evidence.time_period!r}"
            )

            return verdict_from_error(
                claim,
                evidence,
                best_error,
                best_match,
                config,
                _classify_mismatch,
            )

    kosis_unit = evidence.unit or ""

    if not is_same_unit_type(claim_unit, kosis_unit):
        logger.info(f"단위 타입 불일치: claim={claim_unit!r} kosis={kosis_unit!r}")
        return VerificationResult(
            claim_id=claim.claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3,
            evidence=evidence,
        )

    official = normalize_value(evidence.official_value, kosis_unit)
    denom = max(abs(official), abs(claimed), 1e-9)
    diff_pct = abs(claimed - official) / denom * 100

    return verdict_from_error(
        claim, evidence, diff_pct / 100, None, config, _classify_mismatch,
    )


def _primary_year_from_period(text: str | None) -> str | None:
    if not text or not str(text).strip():
        return None
    m = re.search(r"(?:19|20)\d{2}", str(text))
    return m.group(0) if m else None


def _norm_token(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(str(s).split()).lower()


def _population_incompatible(claim_pop: str | None, ev_pop: str | None) -> bool:
    c = _norm_token(claim_pop)
    e = _norm_token(ev_pop)
    if not c or not e:
        return False
    if c in e or e in c:
        return False
    return True


def _classify_mismatch(
    claim: Claim,
    evidence: Evidence,
    diff_pct: float,
    config: dict,
) -> MismatchType:
    """MISMATCH 세부 유형 분류 (fallback 프로필, LLM 미사용)."""
    vconf = config.get("verification", {}) if config else {}
    exaggeration_pct = float(vconf.get("exaggeration_diff_percent", 20.0))

    schema = claim.schema
    if schema is None:
        return (
            MismatchType.EXAGGERATION
            if diff_pct > exaggeration_pct
            else MismatchType.VALUE
        )

    cy = _primary_year_from_period(schema.time_period)
    ey = _primary_year_from_period(evidence.time_period)
    if cy and ey and cy != ey:
        return MismatchType.TIME_PERIOD

    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    ev_pop = raw.get("population") or raw.get("population_label")
    if isinstance(ev_pop, (list, tuple)):
        ev_pop = " ".join(str(x) for x in ev_pop)
    if schema.population and _population_incompatible(schema.population, ev_pop):
        return MismatchType.POPULATION

    if diff_pct > exaggeration_pct:
        return MismatchType.EXAGGERATION

    return MismatchType.VALUE
