"""[리팩] agent Observation / verifier Evidence → decide_verdict 공통 입력 (판정 규칙 없음)"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from structverify.core.schemas import (
    Claim,
    Evidence,
    MismatchType,
    VerificationResult,
    VerdictType,
)
from structverify.utils.logger import get_logger

if TYPE_CHECKING:
    from structverify.graph.claim_graph import ClaimGraph
    from structverify.memory.working_memory import DocumentWorkingMemory

logger = get_logger(__name__)


@dataclass
class NormalizedInput:
    """decide_verdict가 소비하는 fallback 경로 입력."""

    evidence: Evidence
    claim_year: str | None = None
    claim_year_month: str | None = None


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
