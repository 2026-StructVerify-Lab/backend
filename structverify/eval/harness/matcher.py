"""Match golden eval claims to pipeline Claim / VerificationResult."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from structverify.core.schemas import Claim, VerificationResult
from structverify.eval.builder.schemas import EvalClaim
from structverify.eval.builder.text_utils import normalize_claim_text

VERDICT_LABELS = ("match", "mismatch", "unverifiable")


@dataclass
class MatchedClaimRow:
    gold_claim_id: str
    gold_verdict: str
    gold_stat_id: str | None
    gold_claim_text: str
    pipeline_claim_id: str | None
    pipeline_claim_text: str | None
    predicted_verdict: str | None
    predicted_stat_id: str | None
    verdict_correct: bool | None
    stat_id_correct: bool | None


def _match_score(gold_norm: str, pred_norm: str) -> int:
    if not gold_norm or not pred_norm:
        return 0
    if gold_norm == pred_norm:
        return 3
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return 2 + min(len(gold_norm), len(pred_norm)) // 100
    return 0


def match_golden_claims(
    gold_claims: list[EvalClaim],
    pipeline_claims: list[Claim],
    results: list[VerificationResult],
) -> tuple[list[MatchedClaimRow], int]:
    """Greedy 1:1 match gold claims to pipeline claims by normalized text."""
    result_by_claim: dict[UUID, VerificationResult] = {
        r.claim_id: r for r in results
    }
    used_pipeline: set[int] = set()
    rows: list[MatchedClaimRow] = []

    candidates: list[tuple[int, int, int]] = []
    pred_norms = [normalize_claim_text(c.claim_text) for c in pipeline_claims]
    for gi, gold in enumerate(gold_claims):
        gnorm = normalize_claim_text(gold.claim_text)
        for pi, pnorm in enumerate(pred_norms):
            score = _match_score(gnorm, pnorm)
            if score > 0:
                candidates.append((score, gi, pi))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    gold_to_pipe: dict[int, int] = {}
    for _score, gi, pi in candidates:
        if gi in gold_to_pipe or pi in used_pipeline:
            continue
        gold_to_pipe[gi] = pi
        used_pipeline.add(pi)

    for gi, gold in enumerate(gold_claims):
        pi = gold_to_pipe.get(gi)
        pipe_claim: Claim | None = pipeline_claims[pi] if pi is not None else None
        result: VerificationResult | None = None
        predicted_verdict: str | None = None
        predicted_stat_id: str | None = None
        if pipe_claim is not None:
            result = result_by_claim.get(pipe_claim.claim_id)
            if result is not None:
                predicted_verdict = result.verdict.value
                if result.evidence and result.evidence.stat_table_id:
                    predicted_stat_id = result.evidence.stat_table_id

        verdict_correct: bool | None = None
        if pi is not None and predicted_verdict is not None:
            verdict_correct = predicted_verdict == gold.gold_verdict

        stat_id_correct: bool | None = None
        if (
            pi is not None
            and gold.gold_verdict != "unverifiable"
            and gold.gold_stat_id
            and predicted_stat_id
        ):
            stat_id_correct = predicted_stat_id == gold.gold_stat_id

        rows.append(
            MatchedClaimRow(
                gold_claim_id=gold.claim_id,
                gold_verdict=gold.gold_verdict,
                gold_stat_id=gold.gold_stat_id,
                gold_claim_text=gold.claim_text,
                pipeline_claim_id=str(pipe_claim.claim_id) if pipe_claim else None,
                pipeline_claim_text=pipe_claim.claim_text if pipe_claim else None,
                predicted_verdict=predicted_verdict,
                predicted_stat_id=predicted_stat_id,
                verdict_correct=verdict_correct,
                stat_id_correct=stat_id_correct,
            )
        )

    extra_predictions = len(pipeline_claims) - len(used_pipeline)
    return rows, extra_predictions


def article_prediction_record(
    article_id: str,
    intended_domain: str,
    article_scope: str,
    detected_domain: str | None,
    rows: list[MatchedClaimRow],
    extra_predictions: int,
    pipeline_claim_count: int,
    pipeline_result_count: int,
    error: str | None = None,
) -> dict[str, Any]:
    matched = sum(1 for r in rows if r.pipeline_claim_id is not None)
    correct = sum(1 for r in rows if r.verdict_correct is True)
    return {
        "article_id": article_id,
        "intended_domain": intended_domain,
        "article_scope": article_scope,
        "detected_domain": detected_domain,
        "gold_claims": len(rows),
        "matched_claims": matched,
        "verdict_correct": correct,
        "extra_predictions": extra_predictions,
        "pipeline_claim_count": pipeline_claim_count,
        "pipeline_result_count": pipeline_result_count,
        "error": error,
        "claims": [
            {
                "gold_claim_id": r.gold_claim_id,
                "gold_verdict": r.gold_verdict,
                "gold_stat_id": r.gold_stat_id,
                "predicted_verdict": r.predicted_verdict,
                "predicted_stat_id": r.predicted_stat_id,
                "verdict_correct": r.verdict_correct,
                "stat_id_correct": r.stat_id_correct,
                "matched": r.pipeline_claim_id is not None,
            }
            for r in rows
        ],
    }
