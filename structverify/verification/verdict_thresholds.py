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
