"""
detection/claim_detector.py — 검증 가능 주장 탐지 (Step 4)

변경 요약
- 기존: has_numeric=True 문장만 필터 → LLM check-worthiness
- 변경: sentence candidate scoring → 상위 후보만 LLM check-worthiness

즉 Step 4를
1) Sentence Candidate Detection
2) Claim Detection (check-worthiness)
로 분리한다.
"""
from __future__ import annotations

from structverify.core.schemas import Claim, ClaimType, SIRDocument, SourceOffset
from structverify.detection.candidate_scorer import score_candidate
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

CHECK_WORTHY_PROMPT = """팩트체크 전문가로서 아래 문장이 공식 통계로 검증 가능한 수치 기반 주장인지 판별하세요.

기준:
1) 공식 통계 또는 공공 데이터와 연결 가능
2) 단순 일정/발언 소개/감상이 아니라 사실 주장
3) 가능한 경우 claim_type도 분류

문장: "{sentence}"

JSON: {{"is_check_worthy": true/false, "score": 0.0, "claim_type": "increase|decrease|scale|comparison|forecast|null"}}"""


async def detect_claims(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> list[Claim]:
    """
    SIR Tree에서 검증 가능한 주장 탐지.

    단계
    1) sentence candidate scoring
    2) high-score 문장만 check-worthiness 판별
    3) threshold 이상 claim만 Claim 객체로 변환
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    cd_cfg = config.get("candidate_detection", {})
    candidate_threshold = float(cd_cfg.get("threshold", 0.65))
    min_conf = float(config.get("verification", {}).get("min_confidence", 0.7))

    candidate_count = 0
    claims: list[Claim] = []

    for block in sir_doc.blocks:
        for sent in block.sentences:
            # 1) candidate scoring
            score, label, source, signals = await score_candidate(
                sentence=sent.text,
                config=config,
                context={
                    "block_id": block.block_id,
                    "domain": sir_doc.detected_domain,
                },
            )

            sent.candidate_score = score
            sent.candidate_label = label
            sent.candidate_source = source
            sent.candidate_signals = signals

            if score < candidate_threshold or not label:
                continue

            candidate_count += 1

            # 2) LLM check-worthiness
            cw_score, ctype = await _check_worthiness(llm, sent.text)
            if cw_score < min_conf:
                continue

            # 3) Claim 객체 생성
            claims.append(
                Claim(
                    doc_id=sir_doc.doc_id,
                    block_id=block.block_id,
                    sent_id=sent.sent_id,
                    claim_text=sent.text,
                    claim_type=ctype,
                    check_worthy_score=cw_score,
                    graph_anchor_id=sent.graph_anchor_id,
                    source_offset=SourceOffset(
                        char_start=sent.char_offset_start,
                        char_end=sent.char_offset_end,
                        page=block.source_offset.page,
                    ),
                )
            )

    logger.info(f"candidate 문장: {candidate_count}건")
    logger.info(f"검증 가능 주장: {len(claims)}건")
    return claims


async def _check_worthiness(
    llm: LLMClient,
    sentence: str,
) -> tuple[float, ClaimType | None]:
    """
    LLM 기반 check-worthiness 판별.
    candidate detection 이후의 2차 정밀 판별 단계.
    """
    try:
        r = await llm.generate_json(CHECK_WORTHY_PROMPT.format(sentence=sentence))
        is_check_worthy = bool(r.get("is_check_worthy", False))
        score = float(r.get("score", 0.0))
        if not is_check_worthy:
            return 0.0, None

        ct = r.get("claim_type")
        ctype = ClaimType(ct) if ct and ct != "null" else None
        return score, ctype
    except Exception as e:
        logger.error(f"check-worthiness 실패: {e}")
        return 0.0, None