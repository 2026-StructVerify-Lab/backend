"""
detection/claim_detector.py — 검증 가능 주장 탐지 (Step 4)

[김예슬]
- check-worthiness 프롬프트 설계 및 튜닝 담당
- candidate scoring → LLM 2차 판별 구조 담당
- domain-packs 기반 도메인별 프롬프트 주입

[변경 요약]
- 기존: has_numeric=True 문장만 필터 → LLM check-worthiness
- 변경: LLM/학습 기반 sentence candidate scoring → 상위 후보만 LLM check-worthiness

[설계 원칙]
- Regex 필터(has_numeric 등) 로 1차 후보를 결정하지 않습니다.
- candidate_scorer.py의 Teacher LLM이 0~1 점수를 계산하고,
  threshold 이상인 문장만 이 check-worthiness 단계로 전달됩니다.
- 즉, Step 4를 다음 두 단계로 분리합니다:
  1) Sentence Candidate Detection (candidate_scorer.py — Teacher LLM)
  2) Claim Detection / Check-Worthiness (LLM 중량 모델)

[박재윤 - 2026-05-14]: CHECK_WORTHY_PROMPT 개선
   · 예보/예상/전망 수치 → false 기준 명시
   · 순위 표현 단독 → false 기준 명시
   · 외국 기관 발표 수치 → false 기준 명시
   · positive/negative 예시 추가

[박재윤 - 2026-05-18]: CHECK_WORTHY_PROMPT 검증 가능 기준 보강
   · 기준 1번: "공식 통계 연결" → "정부/공공기관 발표 수치" 로 구체화
   · 공시가격 변동률 등 부동산 수치 positive 예시 추가
"""
from __future__ import annotations

import asyncio  # 병렬처리

from structverify.core.schemas import Claim, SIRDocument, SourceOffset
from structverify.detection.candidate_scorer import score_candidate
from structverify.detection._llm import get_llm_client
from structverify.detection.claims.worthiness import _check_worthiness
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def detect_claims(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> list[Claim]:
    """
    SIR Tree에서 검증 가능한 주장 탐지.

    단계
    1) LLM 기반 sentence candidate scoring
    2) high-score 문장만 check-worthiness 판별 (LLM 중량 모델)
    3) threshold 이상 claim만 Claim 객체로 변환

    TODO [김예슬]: claim_type 분류 정확도 개선
      - "increase": 증가/상승/올랐다
      - "decrease": 감소/하락/내렸다
      - "scale": 규모/비율/수준 언급
      - "comparison": A가 B보다 높다/낮다
      - "forecast": 전망/예상/목표
    """
    config = config or {}
    llm = get_llm_client(config)

    cd_cfg = config.get("candidate_detection", {})
    candidate_threshold = float(cd_cfg.get("threshold", 0.65))
    min_conf = float(config.get("verification", {}).get("min_confidence", 0.7))

    # 동시 LLM 호출 수 제한 — HCX 429 rate limit 회피 (config: candidate_detection.concurrency)
    # [2026-05-21] 기본값 5 → 4: HCX-003 burst 시 전체 429 폭주 빈발해서 안전 기본값 하향.
    concurrency = int(cd_cfg.get("concurrency", 4))
    sem = asyncio.Semaphore(concurrency)

    sentence_items = []
    for block in sir_doc.blocks:
        for sent in block.sentences:
            sentence_items.append((block, sent))

    async def score_one(block, sent):
        async with sem:
            score, label, source, signals = await score_candidate(
                sentence=sent.text,
                config=config,
                context={
                    "block_id": block.block_id,
                    "domain": sir_doc.detected_domain,
                },
            )
            return block, sent, score, label, source, signals

    # 1) candidate scoring 병렬 처리
    score_results = await asyncio.gather(
        *[score_one(block, sent) for block, sent in sentence_items],
        return_exceptions=True,
    )

    candidates = []

    for result in score_results:
        if isinstance(result, Exception):
            logger.warning(f"candidate scoring 실패: {result}")
            continue

        block, sent, score, label, source, signals = result

        sent.candidate_score = score
        sent.candidate_label = label
        sent.candidate_source = source
        sent.candidate_signals = signals

        if score >= candidate_threshold and label:
            candidates.append((block, sent))

    logger.info(f"candidate 문장: {len(candidates)}건")

    domain = sir_doc.detected_domain

    async def check_one(block, sent):
        async with sem:
            cw_score, claim_type, canonical_type = await _check_worthiness(
                llm,
                sent.text,
                config=config,
                domain=domain,
            )
            return block, sent, cw_score, claim_type, canonical_type

    # 2) check-worthiness도 병렬 처리
    check_results = await asyncio.gather(
        *[check_one(block, sent) for block, sent in candidates],
        return_exceptions=True,
    )

    claims: list[Claim] = []

    for result in check_results:
        if isinstance(result, Exception):
            logger.warning(f"check-worthiness 실패: {result}")
            continue

        block, sent, cw_score, claim_type, canonical_type = result

        if cw_score < min_conf:
            continue

        claims.append(
            Claim(
                doc_id=sir_doc.doc_id,
                block_id=block.block_id,
                sent_id=sent.sent_id,
                claim_text=sent.text,
                claim_type=claim_type,
                canonical_type=canonical_type,
                check_worthy_score=cw_score,
                graph_anchor_id=sent.graph_anchor_id,
                source_offset=SourceOffset(
                    char_start=sent.char_offset_start,
                    char_end=sent.char_offset_end,
                    page=block.source_offset.page if block.source_offset else None,
                ),
            )
        )
    logger.info(f"검증 가능 주장: {len(claims)}건")
    return claims
