"""[리팩] explainer._fallback_explanation 분리 — LLM 실패 시 기본 문구 생성"""
from __future__ import annotations

from structverify.core.schemas import Claim, VerdictType, VerificationResult


def _fallback_explanation(claim: Claim, result: VerificationResult) -> str:
    """LLM 실패 시 기본 텍스트로 fallback."""
    verdict_kr = {
        VerdictType.MATCH: "일치",
        VerdictType.MISMATCH: "불일치",
        VerdictType.UNVERIFIABLE: "검증 불가",
    }.get(result.verdict, result.verdict.value)

    base = f'"{claim.claim_text[:40]}..." — 판정: {verdict_kr}'

    if result.verdict == VerdictType.MISMATCH and result.evidence:
        ev = result.evidence
        schema = claim.schema
        if schema and schema.value and ev.official_value:
            base += (
                f" | 기사: {schema.value}{schema.unit or ''}"
                f" / 공식: {ev.official_value}{ev.unit or ''}"
            )

    if result.provenance_summary:
        base += f" | {result.provenance_summary}"

    return base
