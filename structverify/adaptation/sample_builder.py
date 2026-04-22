"""
adaptation/sample_builder.py — 학습 샘플 자동 생성

두 가지 모드를 지원한다:
  - mode="pretrain"  : 합성 데이터(synthetic_generator) → LoRA 사전학습 포맷
  - mode="finetune"  : 운영 피드백(feedback_store) → LoRA 추가학습 포맷

[참고] Self-Instruct (Wang et al., ACL 2023)
  - https://github.com/yizhongw/self-instruct
  - 합성 데이터를 instruction-tuning 포맷으로 변환

[참고] KnowLA (NAACL 2024)
  - KG 정보를 PEFT adapter에 반영하는 학습 데이터 구성 전략
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
    학습 데이터를 LoRA 학습 포맷으로 변환한다.

    Args:
        events: 피드백 이벤트 리스트 (mode="finetune"일 때)
        synthetic: 합성 데이터 리스트 (mode="pretrain"일 때)
        mode: "pretrain" | "finetune"

    Returns:
        LoRA 학습 가능한 (input, output) 쌍 리스트
    """
    if mode == "pretrain":
        return _build_pretrain_samples(synthetic or [])
    elif mode == "finetune":
        return _build_finetune_samples(events or [])
    else:
        raise ValueError(f"미지원 모드: {mode}")


# ═══════════════════════════════════════
# 사전학습 (합성 데이터 → 학습 포맷)
# ═══════════════════════════════════════

def _build_pretrain_samples(
    synthetic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    합성 데이터 → LoRA 학습 포맷 변환. 3가지 태스크 동시 생성:
      1) claim_to_stat: 주장 → 관련 통계표 매핑
      2) claim_to_schema: 주장 → 구조화 스키마 추출
      3) stat_to_claim: 통계표 → 검증 가능 주장 판별
    """
    samples: list[dict[str, Any]] = []

    for item in synthetic:
        claim = item.get("claim", "")
        stat_id = item.get("stat_id", "")
        stat_name = item.get("stat_name", "")
        indicator = item.get("indicator", "")
        schema = item.get("schema", {})

        if not claim or not stat_id:
            continue

        # 태스크 1: 주장 → 통계표 매핑
        samples.append({
            "task": "claim_to_stat",
            "input": f"아래 주장을 검증하기 위해 필요한 KOSIS 통계표를 찾으세요.\n주장: {claim}",
            "output": json.dumps(
                {"stat_id": stat_id, "stat_name": stat_name, "indicator": indicator},
                ensure_ascii=False),
        })

        # 태스크 2: 주장 → 스키마 추출
        if schema:
            samples.append({
                "task": "claim_to_schema",
                "input": f"아래 주장에서 검증에 필요한 핵심 정보를 추출하세요.\n주장: {claim}",
                "output": json.dumps(schema, ensure_ascii=False),
            })

        # 태스크 3: 통계표 → 주장 판별 (역방향)
        samples.append({
            "task": "stat_to_claim",
            "input": f'통계표 "{stat_name}"({stat_id})로 아래 주장을 검증할 수 있습니까?\n주장: {claim}',
            "output": '{"verifiable": true}',
        })

    logger.info(f"사전학습 샘플: {len(samples)}건 (원본 {len(synthetic)}건)")
    return samples


# ═══════════════════════════════════════
# 피드백 기반 학습 (기존)
# ═══════════════════════════════════════

def _build_finetune_samples(
    events: list[FeedbackEvent],
) -> list[dict[str, Any]]:
    """
    피드백 이벤트 → 학습용 (input, label) 쌍 생성 (기존 로직)

    TODO: Claim Detection, Schema Induction, Explanation 태스크 추가
    """
    samples: list[dict[str, Any]] = []
    for ev in events:
        samples.append({
            "task": "verdict_correction",
            "input": str(ev.claim_id),
            "original": ev.original_verdict.value if ev.original_verdict else None,
            "corrected": ev.corrected_verdict.value if ev.corrected_verdict else None,
            "note": ev.reviewer_note,
        })
    logger.info(f"피드백 샘플: {len(samples)}건 (원본 {len(events)}건)")
    return samples
