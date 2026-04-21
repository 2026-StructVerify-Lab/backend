"""
detection/claim_detector.py — 검증 가능 주장 탐지 (Step 4)

[참고] ClaimBuster (Hassan et al., VLDB 2017) — https://github.com/idirlab/claimBuster
  check-worthy claim 탐지 기준 설계에 참고. 본 프로젝트에서는 LLM으로 대체.

[참고] FActScore (Min et al., EMNLP 2023) — https://github.com/shmsw25/FActScore
  긴 텍스트에서 atomic fact 단위로 분해하여 독립 검증하는 granularity 설계에 참고.
"""
from __future__ import annotations
from structverify.core.schemas import Claim, ClaimType, SIRDocument, SourceOffset
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

CHECK_WORTHY_PROMPT = """팩트체크 전문가로서 아래 문장이 공식 통계로 검증 가능한 수치 기반 주장인지 판별하세요.

기준: 1) 구체적 수치 포함 2) 공식 통계로 검증 가능 3) 의견/추측이 아닌 팩트

문장: "{sentence}"

JSON: {{"is_check_worthy": true/false, "score": 0.0~1.0, "claim_type": "increase|decrease|scale|comparison|forecast|null"}}"""


async def detect_claims(sir_doc: SIRDocument, config: dict | None = None) -> list[Claim]:
    """
    SIR Tree에서 검증 가능 주장 탐지.
    1) has_numeric=True 문장 필터 → 2) LLM check-worthiness → 3) 점수 기반 선별
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))
    min_conf = config.get("verification", {}).get("min_confidence", 0.7)

    candidates = [(b.block_id, s, b) for b in sir_doc.blocks for s in b.sentences if s.has_numeric]
    logger.info(f"수치 포함 문장: {len(candidates)}건")

    claims: list[Claim] = []
    for block_id, sent, block in candidates:
        score, ctype = await _check_worthiness(llm, sent.text)
        if score >= min_conf:
            claims.append(Claim(
                doc_id=sir_doc.doc_id, block_id=block_id, sent_id=sent.sent_id,
                claim_text=sent.text, claim_type=ctype, check_worthy_score=score,
                graph_anchor_id=sent.graph_anchor_id,
                source_offset=SourceOffset(
                    char_start=sent.char_offset_start, char_end=sent.char_offset_end,
                    page=block.source_offset.page),
            ))
    logger.info(f"검증 가능 주장: {len(claims)}건")
    return claims


async def _check_worthiness(llm: LLMClient, sentence: str) -> tuple[float, ClaimType | None]:
    """LLM 기반 check-worthiness 판별 — ClaimBuster SVM을 LLM zero-shot으로 대체"""
    try:
        r = await llm.generate_json(CHECK_WORTHY_PROMPT.format(sentence=sentence))
        score = float(r.get("score", 0.0))
        ct = r.get("claim_type")
        ctype = ClaimType(ct) if ct and ct != "null" else None
        return score, ctype
    except Exception as e:
        logger.error(f"check-worthiness 실패: {e}")
        return 0.0, None
