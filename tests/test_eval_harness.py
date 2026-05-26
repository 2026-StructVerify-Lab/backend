"""Tests for eval harness matcher, metrics, and visualization."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from structverify.core.schemas import (
    Claim,
    ClaimSchema,
    Evidence,
    SourceOffset,
    VerdictType,
    VerificationResult,
)
from structverify.eval.builder.schemas import EvalClaim, GoldSchema
from structverify.eval.harness.matcher import match_golden_claims
from structverify.eval.harness.metrics import compute_report
from structverify.eval.harness.visualize import render_eval_report


def _pipe_claim(text: str) -> Claim:
    return Claim(
        doc_id=uuid4(),
        block_id="b1",
        sent_id="s1",
        claim_text=text,
        source_offset=SourceOffset(),
    )


def _gold(claim_id: str, text: str, verdict: str = "match") -> EvalClaim:
    return EvalClaim(
        claim_id=claim_id,
        claim_text=text,
        gold_verdict=verdict,  # type: ignore[arg-type]
        gold_stat_id="DT_TEST" if verdict != "unverifiable" else None,
        gold_schema=GoldSchema(indicator="x", value=1.0, time_period="2024"),
    )


def test_match_exact_claim_text():
    gold = [_gold("g1", "2024년 GDP는 2.6%였다.")]
    pipe = [_pipe_claim("2024년 GDP는 2.6%였다.")]
    result = VerificationResult(
        claim_id=pipe[0].claim_id,
        verdict=VerdictType.MATCH,
        evidence=Evidence(source_name="kosis", stat_table_id="DT_TEST"),
    )
    rows, extra = match_golden_claims(gold, pipe, [result])
    assert extra == 0
    assert len(rows) == 1
    assert rows[0].pipeline_claim_id is not None
    assert rows[0].verdict_correct is True
    assert rows[0].stat_id_correct is True


def test_match_substring_partial():
    gold = [_gold("g1", "출생아 수는 2만 명")]
    pipe = [_pipe_claim("올해 출생아 수는 2만 명을 넘겼다.")]
    result = VerificationResult(
        claim_id=pipe[0].claim_id,
        verdict=VerdictType.MISMATCH,
    )
    rows, extra = match_golden_claims(gold, pipe, [result])
    assert rows[0].pipeline_claim_id is not None
    assert rows[0].verdict_correct is False


def test_match_one_to_one_no_duplicate():
    gold = [
        _gold("g1", "첫 번째 문장입니다."),
        _gold("g2", "두 번째 다른 문장입니다."),
    ]
    pipe = [
        _pipe_claim("두 번째 다른 문장입니다."),
        _pipe_claim("첫 번째 문장입니다."),
    ]
    results = [
        VerificationResult(claim_id=pipe[0].claim_id, verdict=VerdictType.MATCH),
        VerificationResult(claim_id=pipe[1].claim_id, verdict=VerdictType.MATCH),
    ]
    rows, extra = match_golden_claims(gold, pipe, results)
    assert len(rows) == 2
    assert all(r.pipeline_claim_id for r in rows)
    assert extra == 0


def test_unmatched_gold_missed_extraction():
    gold = [_gold("g1", "없는 문장")]
    rows, extra = match_golden_claims(gold, [], [])
    assert rows[0].pipeline_claim_id is None
    assert rows[0].predicted_verdict is None
    assert extra == 0


def test_compute_report_metrics():
    records = [
        {
            "article_id": "a1",
            "intended_domain": "economy",
            "gold_claims": 2,
            "matched_claims": 2,
            "verdict_correct": 1,
            "claims": [
                {
                    "gold_verdict": "match",
                    "predicted_verdict": "match",
                    "matched": True,
                    "stat_id_correct": True,
                },
                {
                    "gold_verdict": "mismatch",
                    "predicted_verdict": "match",
                    "matched": True,
                    "stat_id_correct": False,
                },
            ],
        },
        {
            "article_id": "a2",
            "intended_domain": "economy",
            "gold_claims": 1,
            "matched_claims": 0,
            "verdict_correct": 0,
            "claims": [
                {
                    "gold_verdict": "match",
                    "predicted_verdict": None,
                    "matched": False,
                },
            ],
        },
    ]
    report = compute_report("test_set", "test_run", records)
    assert report["claims_gold"] == 3
    assert report["claims_matched"] == 2
    assert report["summary"]["extraction_recall"] == pytest.approx(2 / 3)
    assert report["summary"]["verdict_accuracy"] == pytest.approx(0.5)
    assert "confusion_matrix" in report
    assert report["per_domain"]["economy"]["gold"] == 3


def test_render_eval_report_smoke(tmp_path: Path):
    report = {
        "dataset_id": "t",
        "run_id": "t_run",
        "articles_total": 1,
        "claims_gold": 1,
        "summary": {
            "verdict_accuracy": 1.0,
            "extraction_recall": 1.0,
            "stat_id_accuracy": 0.0,
        },
        "per_verdict": {
            "match": {"precision": 1, "recall": 1, "f1": 1, "support": 1},
            "mismatch": {"precision": 0, "recall": 0, "f1": 0, "support": 0},
            "unverifiable": {"precision": 0, "recall": 0, "f1": 0, "support": 0},
        },
        "confusion_matrix": {
            "labels": ["match", "mismatch", "unverifiable"],
            "matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        },
        "per_domain": {
            "economy": {"accuracy": 1.0, "extraction_recall": 1.0, "matched": 1, "gold": 1},
        },
        "per_article": [{"article_id": "a1", "accuracy": 1.0, "matched": 1, "gold": 1}],
    }
    (tmp_path / "report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    ok = render_eval_report(tmp_path)
    if ok:
        assert (tmp_path / "report.html").exists()
        assert (tmp_path / "charts" / "verdict_accuracy.png").exists()
    # matplotlib optional — no failure if missing
