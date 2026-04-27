"""
adaptation/sample_builder.py — 학습 샘플 포맷 변환

[김예슬 - 2026-04-24]
- candidate_detection 태스크 샘플 포맷 추가
  · instruction/input/output 필드 구조로 통일
  · HuggingFace Dataset 호환 + NCP Tuning API JSONL 호환
- 태스크별 샘플 수 로깅 추가

[참고] Self-Instruct (Wang et al., ACL 2023)
[참고] KnowLA (NAACL 2024)
"""
from __future__ import annotations

import json
from typing import Any

from structverify.core.schemas import FeedbackEvent
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def build_training_samples(
    events: list[FeedbackEvent] | None = None,
    synthetic: list[dict[str, Any]] | None = None,
    mode: str = "finetune",
) -> list[dict[str, Any]]:
    """
    학습 데이터 → LoRA 학습 포맷 변환.

    Args:
        events: 피드백 이벤트 (mode="finetune")
        synthetic: 합성 데이터 (mode="pretrain")
        mode: "pretrain" | "finetune"

    Returns:
        [{"task": ..., "instruction": ..., "input": ..., "output": ...}, ...]
    """
    if mode == "pretrain":
        return _build_pretrain_samples(synthetic or [])
    elif mode == "finetune":
        return _build_finetune_samples(events or [])
    raise ValueError(f"미지원 모드: {mode}")


def _build_pretrain_samples(synthetic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    합성 데이터 → instruction-tuning 포맷 변환.

    생성 태스크:
      1) claim_to_stat      : 주장 → 관련 통계표
      2) claim_to_schema    : 주장 → 구조화 스키마
      3) stat_to_claim      : 통계표 → 주장 판별 (역방향)
      4) candidate_detection: 검증 후보/비후보 분류
    """
    samples: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}

    for item in synthetic:
        task = item.get("task", "")

        # ── candidate detection 샘플 ──────────────────────────────────
        if task == "candidate_detection":
            sentence = item.get("sentence", "")
            label    = item.get("candidate_label", False)

            sample = {
                "task": "candidate_detection",
                "instruction": (
                    "아래 문장이 공식 통계로 검증 가능한 수치 기반 주장인지 판단하세요.\n"
                    'JSON으로만 답하세요: {"candidate_label": true 또는 false}'
                ),
                "input":  sentence,
                "output": json.dumps({"candidate_label": label}, ensure_ascii=False),
            }
            samples.append(sample)
            task_counts["candidate_detection"] = task_counts.get("candidate_detection", 0) + 1
            continue

        # ── claim/schema 샘플 ─────────────────────────────────────────
        claim     = item.get("claim", "")
        stat_id   = item.get("stat_id", "")
        stat_name = item.get("stat_name", "")
        indicator = item.get("indicator", "")
        schema    = item.get("schema", {})

        if not claim or not stat_id:
            continue

        # 태스크 1: claim_to_stat
        samples.append({
            "task": "claim_to_stat",
            "instruction": "아래 주장을 검증하기 위해 필요한 KOSIS 통계표를 찾으세요.",
            "input":  claim,
            "output": json.dumps(
                {"stat_id": stat_id, "stat_name": stat_name, "indicator": indicator},
                ensure_ascii=False,
            ),
        })
        task_counts["claim_to_stat"] = task_counts.get("claim_to_stat", 0) + 1

        # 태스크 2: claim_to_schema
        if schema and isinstance(schema, dict) and "raw" not in schema:
            samples.append({
                "task": "claim_to_schema",
                "instruction": "아래 주장에서 검증에 필요한 핵심 정보를 추출하세요.",
                "input":  claim,
                "output": json.dumps(schema, ensure_ascii=False),
            })
            task_counts["claim_to_schema"] = task_counts.get("claim_to_schema", 0) + 1

        # 태스크 3: stat_to_claim (역방향)
        samples.append({
            "task": "stat_to_claim",
            "instruction": (
                f'통계표 "{stat_name}"({stat_id})로 아래 주장을 검증할 수 있습니까?\n'
                'JSON으로만 답하세요: {"verifiable": true 또는 false}'
            ),
            "input":  claim,
            "output": json.dumps({"verifiable": True}, ensure_ascii=False),
        })
        task_counts["stat_to_claim"] = task_counts.get("stat_to_claim", 0) + 1

    logger.info(f"사전학습 샘플: 총 {len(samples)}건")
    logger.info(f"태스크별 분포: {task_counts}")
    return samples


def _build_finetune_samples(events: list[FeedbackEvent]) -> list[dict[str, Any]]:
    """피드백 이벤트 → 추가 학습 포맷 변환"""
    samples: list[dict[str, Any]] = []
    for ev in events:
        if not ev.corrected_verdict:
            continue
        samples.append({
            "task": "verdict_correction",
            "instruction": "아래 주장에 대한 검증 결과를 판정하세요.",
            "input":  str(ev.claim_id),
            "output": json.dumps(
                {"verdict": ev.corrected_verdict.value},
                ensure_ascii=False,
            ),
            "original_verdict": ev.original_verdict.value if ev.original_verdict else None,
            "reviewer_note":    ev.reviewer_note,
        })

    logger.info(f"피드백 샘플: {len(samples)}건 (원본 {len(events)}건)")
    return samples