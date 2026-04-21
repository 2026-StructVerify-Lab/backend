"""
explanation/explainer.py — LLM 기반 설명 + Provenance 렌더링 (Step 9)

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
  Agent의 최종 Observation 단계에서 판정 근거를 자연어로 설명
"""
from __future__ import annotations
from structverify.core.schemas import Claim, VerificationResult
from structverify.graph.provenance import render_provenance_text
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

EXPLANATION_PROMPT = """팩트체크 결과를 설명하세요.

주장: "{claim_text}"
판정: {verdict}
기사 수치: {claimed_value}
공식 수치: {official_value}
출처: {provenance}

다음 내용을 포함하여 한국어로 설명하세요:
1) 어떤 통계를 근거로 사용했는지
2) 어떤 방식으로 비교했는지
3) 판정 사유
4) 한계점"""


async def generate_explanation(claim: Claim, result: VerificationResult,
                                config: dict | None = None) -> str:
    """검증 결과에 대한 자연어 설명 + Provenance 렌더링"""
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    prov_text = ""
    if result.evidence and result.evidence.provenance:
        prov_text = render_provenance_text(result.evidence.provenance)
        result.provenance_summary = prov_text

    prompt = EXPLANATION_PROMPT.format(
        claim_text=claim.claim_text, verdict=result.verdict.value,
        claimed_value=claim.schema.value if claim.schema else "N/A",
        official_value=result.evidence.official_value if result.evidence else "N/A",
        provenance=prov_text or "N/A",
    )

    try:
        explanation = await llm.generate(prompt)
        return explanation
    except Exception as e:
        logger.error(f"설명 생성 실패: {e}")
        return f"설명 생성 실패: {e}"
