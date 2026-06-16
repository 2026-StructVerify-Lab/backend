"""[리팩] explainer의 LLMClient 호출 → Step 9 전용 thin wrapper"""
from __future__ import annotations

from structverify.utils.llm_client import LLMClient

from ._config import get_explanation_settings

_EXPLANATION_SYSTEM_PROMPT = (
    "팩트체크 전문 작가. 명확하고 간결한 한국어로 작성하세요."
)


async def generate_explanation_text(
    prompt: str,
    config: dict | None = None,
) -> str:
    """검증 설명용 LLM 텍스트 생성."""
    expl_cfg = get_explanation_settings(config)
    model_tier = expl_cfg.get("model_tier", "heavy")
    llm = LLMClient(config=(config or {}).get("llm", {}))
    return await llm.generate(
        prompt=prompt,
        system_prompt=_EXPLANATION_SYSTEM_PROMPT,
        model_tier=model_tier,
    )
