"""
verification/verifier.py — Deterministic Verification Engine (Step 8)

수치 비교는 LLM이 아닌 deterministic engine이 수행 (hallucination 방지).

[참고] FEVER (Thorne et al., NAACL 2018)
  SUPPORTS/REFUTES/NEI 3단계 판정 → match/mismatch/unverifiable 매핑
"""
from __future__ import annotations
import math
from structverify.core.schemas import (
    Claim, Evidence, VerificationResult, VerdictType, MismatchType)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def verify_claim(claim: Claim, evidence: Evidence | None,
                 config: dict | None = None) -> VerificationResult:
    """
    공식 통계와 기사 수치를 비교하여 판정.
    검증 자체는 deterministic — LLM 개입 없음.
    """
    config = config or {}
    tolerance = config.get("verification", {}).get("tolerance_percent", 1.0)

    if evidence is None or evidence.official_value is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE, confidence=0.3,
            evidence=evidence)

    claimed = claim.schema.value if claim.schema else None
    if claimed is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE, confidence=0.2,
            evidence=evidence)

    official = evidence.official_value
    diff_pct = abs(claimed - official) / max(abs(official), 1e-9) * 100

    if diff_pct <= tolerance:
        verdict, mtype, conf = VerdictType.MATCH, None, min(0.95, 1.0 - diff_pct / 100)
    else:
        verdict, conf = VerdictType.MISMATCH, min(0.9, diff_pct / 100)
        mtype = _classify_mismatch(claim, evidence)

    result = VerificationResult(
        claim_id=claim.claim_id, verdict=verdict, confidence=conf,
        evidence=evidence, mismatch_type=mtype)
    logger.info(f"검증 결과: {verdict.value} (차이: {diff_pct:.2f}%)")
    return result


def _classify_mismatch(claim: Claim, evidence: Evidence) -> MismatchType:
    """불일치 유형 분류"""
    # TODO: 시점/모집단/과장 불일치 세분화 로직 구현
    return MismatchType.VALUE
