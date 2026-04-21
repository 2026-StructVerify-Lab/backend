"""
detection/domain_classifier.py — 도메인 자동 분류 (Step 3)

입력 텍스트의 도메인을 판별하고 적절한 Domain Pack을 선택한다.

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
  Agent의 첫 단계로 도메인을 판별하여 이후 전략을 결정하는 패턴
"""
from __future__ import annotations
from structverify.core.schemas import SIRDocument
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

DOMAIN_CLASSIFY_PROMPT = """아래 문서의 도메인을 분류하세요.
가능한 도메인: news_economy, news_society, finance, policy, agriculture, health, other

문서 첫 500자:
"{text_preview}"

JSON으로 답하세요: {{"domain": "...", "confidence": 0.0~1.0}}"""


async def classify_domain(sir_doc: SIRDocument, config: dict | None = None) -> str:
    """
    SIR 문서의 도메인을 분류한다.
    결과를 sir_doc.detected_domain에 기록하고 반환한다.

    TODO: 도메인 분류 모델 (fine-tuned classifier) 도입
    TODO: Domain Pack 레지스트리 조회하여 후보 선택
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))
    preview = " ".join(b.content or "" for b in sir_doc.blocks[:5])[:500]

    try:
        result = await llm.generate_json(
            DOMAIN_CLASSIFY_PROMPT.format(text_preview=preview),
            system_prompt="도메인 분류 전문가. JSON만 답하세요.")
        domain = result.get("domain", "other")
    except Exception as e:
        logger.error(f"도메인 분류 실패: {e}")
        domain = "other"

    sir_doc.detected_domain = domain
    logger.info(f"도메인 분류: {domain}")
    return domain
