"""detection/claims/worthiness.py — check-worthiness LLM 판별.

claim_detector.py에서 분리 (로직 move-only).

TODO [김예슬]: 오류 응답 처리 강화 (재시도, score 클램핑)
"""
from __future__ import annotations

from structverify.core.schemas import ClaimType
from structverify.detection.prompts.claim_worthiness import CHECK_WORTHY_PROMPT
from structverify.detection.prompts_loader import resolve_prompt_for_step
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def _check_worthiness(
    llm: LLMClient,
    sentence: str,
    *,
    config: dict | None = None,
    domain: str | None = None,
) -> tuple[float, str | None, ClaimType | None]:
    """
    LLM 기반 check-worthiness 판별 (2차 정밀 판별).
    candidate detection 이후 상위 후보에만 적용.
    """
    try:
        base = CHECK_WORTHY_PROMPT.format(sentence=sentence)
        prompt = resolve_prompt_for_step(
            base, domain, config, step="claim_worthiness",
        )
        r = await llm.generate_json(
            prompt,
            system_prompt="팩트체크 check-worthiness classifier. 반드시 JSON만 출력하세요.",
        )

        is_check_worthy = bool(r.get("is_check_worthy", False))
        score = float(r.get("score", 0.0) or 0.0)

        # true인데 score=0으로 오는 문제 방어
        if is_check_worthy and score <= 0.0:
            score = 0.8

        score = max(0.0, min(score, 1.0))

        if not is_check_worthy:
            return 0.0, None, None
        raw_type = r.get("claim_type")
        canonical = r.get("canonical_type")

        claim_type = raw_type if raw_type and raw_type != "null" else None

        try:
            canonical_type = ClaimType(canonical) if canonical else None
        except ValueError:
            canonical_type = None

        return score, claim_type, canonical_type

    except Exception as e:
        logger.error(f"check-worthiness 실패: {e}")
        return 0.0, None, None
