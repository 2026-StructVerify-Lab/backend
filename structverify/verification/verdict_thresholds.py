"""[리팩] fallback 프로필 오차 구간 판정 — verifier._verdict_from_error 분리"""
from __future__ import annotations

import re
from collections.abc import Callable

from structverify.core.schemas import (
    Claim,
    Evidence,
    MismatchType,
    VerificationResult,
    VerdictType,
)
from structverify.utils.logger import get_logger

from ._config import get_verification_settings

logger = get_logger(__name__)

_THRESHOLD_GTE_KEYWORDS = (
    "넘기", "넘어", "넘는", "넘은", "넘었", "돌파", "초과",
    "이상", "웃돌", "상회", "넘게",
)
_THRESHOLD_LTE_KEYWORDS = (
    "미만", "이하", "밑돌", "하회", "못 미", "못미", "안 되", "안되",
)
_INCREASE_SFX = ("증가율", "상승률")
_DECREASE_SFX = ("감소율", "하락률")


def _agent_thresholds(config: dict | None) -> dict[str, float]:
    settings = get_verification_settings(config, "agent")
    return {
        "value_match_tolerance": float(settings.get("value_match_tolerance", 0.05)),
        "growth_rate_match_pp": float(settings.get("growth_rate_match_pp", 1.5)),
        "growth_rate_unverifiable_pp": float(
            settings.get("growth_rate_unverifiable_pp", 5.0)
        ),
        "difference_rel_tolerance": float(
            settings.get("difference_rel_tolerance", 0.10)
        ),
        "difference_min_tolerance": float(
            settings.get("difference_min_tolerance", 0.02)
        ),
        "difference_unverifiable_multiplier": float(
            settings.get("difference_unverifiable_multiplier", 3.0)
        ),
        "calculate_simple_tolerance": float(
            settings.get("calculate_simple_tolerance", 0.01)
        ),
    }


def classify_atomic_ratio_agent(
    diff_ratio: float,
    config: dict | None = None,
) -> tuple[VerdictType, float]:
    """loop 일반 수치 비교 — value_match_tolerance (기본 5%)."""
    tol = _agent_thresholds(config)["value_match_tolerance"]
    if diff_ratio < tol:
        return VerdictType.MATCH, 0.85
    return VerdictType.MISMATCH, 0.7


def classify_growth_rate_pp_agent(
    diff_pp: float,
    config: dict | None = None,
) -> tuple[VerdictType, float]:
    """loop 증가율 %p 구간 — ≤1.5 MATCH, ≤5 UNVERIFIABLE."""
    t = _agent_thresholds(config)
    if diff_pp <= t["growth_rate_match_pp"]:
        return VerdictType.MATCH, 0.8
    if diff_pp <= t["growth_rate_unverifiable_pp"]:
        return VerdictType.UNVERIFIABLE, 0.4
    return VerdictType.MISMATCH, 0.7


def classify_difference_gap_agent(
    gap: float,
    claimed_diff: float,
    config: dict | None = None,
) -> tuple[VerdictType, float]:
    """loop 차이값 — tol=max(|claimed|×10%, min_tol)."""
    t = _agent_thresholds(config)
    tol = max(abs(claimed_diff) * t["difference_rel_tolerance"], t["difference_min_tolerance"])
    if gap <= tol:
        return VerdictType.MATCH, 0.8
    if gap <= tol * t["difference_unverifiable_multiplier"]:
        return VerdictType.UNVERIFIABLE, 0.4
    return VerdictType.MISMATCH, 0.7


def classify_calculate_simple_agent(
    diff_ratio: float,
    config: dict | None = None,
) -> tuple[VerdictType, float]:
    """loop calculate 일반 수치 — 기본 1% (5%와 별도 유지)."""
    tol = _agent_thresholds(config)["calculate_simple_tolerance"]
    if diff_ratio < tol:
        return VerdictType.MATCH, 0.8
    return VerdictType.MISMATCH, 0.7


def detect_threshold_direction(claim: Claim) -> str | None:
    """부등식 주장 방향 — gte / lte / None (loop._detect_threshold_direction)."""
    text = claim.claim_text or ""
    modifier = ""
    if claim.schema is not None:
        modifier = (claim.schema.modifier or "")
    haystack = f"{text} {modifier}"

    has_gte = any(kw in haystack for kw in _THRESHOLD_GTE_KEYWORDS)
    has_lte = any(kw in haystack for kw in _THRESHOLD_LTE_KEYWORDS)
    if has_gte and has_lte:
        return None
    if has_gte:
        return "gte"
    if has_lte:
        return "lte"
    return None


def growth_rate_direction_mismatch(
    indicator: str,
    calc_rate: float,
) -> bool:
    """증가/감소 방향 불일치 — loop 부호 가드."""
    ind = (indicator or "").strip()
    expects_inc = any(ind.endswith(s) for s in _INCREASE_SFX)
    expects_dec = any(ind.endswith(s) for s in _DECREASE_SFX)
    return (expects_inc and calc_rate < 0) or (expects_dec and calc_rate > 0)


def _fallback_thresholds(config: dict | None) -> dict[str, float]:
    settings = get_verification_settings(config, "fallback")
    return {
        "match_max_error": float(settings.get("match_max_error", 0.10)),
        "unverifiable_max_error": float(settings.get("unverifiable_max_error", 0.30)),
        "mismatch_max_error": float(settings.get("mismatch_max_error", 0.90)),
    }


def classify_error_rate_fallback(
    error_rate: float,
    config: dict | None = None,
) -> tuple[VerdictType, float, bool]:
    """오차율 → (verdict, confidence, mismatch_type_필요여부).

  fallback v7 구간:
    ≤ match_max        → MATCH
    ≤ unverifiable_max → UNVERIFIABLE
    > mismatch_max     → UNVERIFIABLE (표 매칭 오류 의심)
    그 외              → MISMATCH
    """
    t = _fallback_thresholds(config)
    diff_pct = error_rate * 100

    if error_rate <= t["match_max_error"]:
        return VerdictType.MATCH, min(0.95, 1.0 - error_rate), False

    if error_rate <= t["unverifiable_max_error"]:
        logger.info(
            f"검증 결과: unverifiable (오차: {diff_pct:.1f}% — 유사하나 확신 없음)"
        )
        return VerdictType.UNVERIFIABLE, 0.4, False

    if error_rate > t["mismatch_max_error"]:
        logger.info(
            f"검증 결과: unverifiable (오차: {diff_pct:.1f}% — 테이블 매칭 오류 의심)"
        )
        return VerdictType.UNVERIFIABLE, 0.3, False

    return VerdictType.MISMATCH, min(0.9, error_rate), True


def verdict_from_error(
    claim: Claim,
    evidence: Evidence,
    error_rate: float,
    best_match: dict | None,
    config: dict,
    classify_mismatch: Callable[[Claim, Evidence, float, dict], MismatchType],
) -> VerificationResult:
    """오차율 → VerificationResult (fallback 프로필).

    [v3] factcheck_test.py v7 구간 — config.verification.profiles.fallback
    [v6.15] 시점 미상 + tier3 best_match → UNVERIFIABLE
    """
    diff_pct = error_rate * 100

    schema_tp = (
        claim.schema.time_period if claim.schema and claim.schema.time_period else ""
    )
    has_year = bool(re.search(r"\d{4}", schema_tp))
    matched_tier3 = bool(best_match and best_match.get("_tier") == 3)
    if not has_year and matched_tier3:
        logger.info(
            f"검증 결과: unverifiable "
            f"(오차: {diff_pct:.1f}% — 시점 미상 + 시점 매칭 실패, 가짜 매칭 위험) "
            f"→ 엉뚱한 evidence 제거"
        )
        return VerificationResult(
            claim_id=claim.claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.25,
            evidence=None,
        )

    verdict, conf, need_mtype = classify_error_rate_fallback(error_rate, config)
    mtype: MismatchType | None = None

    if verdict == VerdictType.MATCH:
        logger.info(f"검증 결과: match (오차: {diff_pct:.1f}%)")
    elif verdict == VerdictType.MISMATCH and need_mtype:
        mtype = classify_mismatch(claim, evidence, diff_pct, config)
        logger.info(f"검증 결과: mismatch (오차: {diff_pct:.1f}%)")

    return VerificationResult(
        claim_id=claim.claim_id,
        verdict=verdict,
        confidence=conf,
        evidence=evidence,
        mismatch_type=mtype,
    )
