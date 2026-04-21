"""
adaptation/sample_builder.py — 학습 샘플 자동 생성

[참고] KnowLA (NAACL 2024)
  KG 정보를 PEFT adapter에 반영하는 학습 데이터 구성 전략
"""
from __future__ import annotations
from typing import Any
from structverify.core.schemas import FeedbackEvent


def build_training_samples(events: list[FeedbackEvent]) -> list[dict[str, Any]]:
    """
    피드백 이벤트 → 학습용 (input, label) 쌍 생성

    TODO: 태스크별 샘플 포맷 구현
    - Claim Detection: (문장, 0/1)
    - Schema Induction: (문장, JSON)
    - Verdict + Explanation: (입력, 판정, 설명문)
    - Reviewer Correction Pairs: (원판정, 수정판정)
    """
    samples = []
    for ev in events:
        samples.append({
            "task": "verdict_correction",
            "input": str(ev.claim_id),
            "original": ev.original_verdict.value if ev.original_verdict else None,
            "corrected": ev.corrected_verdict.value if ev.corrected_verdict else None,
            "note": ev.reviewer_note,
        })
    return samples
