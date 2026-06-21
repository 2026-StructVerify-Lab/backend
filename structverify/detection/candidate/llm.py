"""detection/candidate/llm.py — candidate scoring Teacher LLM.

candidate_scorer.py에서 분리 (로직 move-only).

TODO [김예슬]: domain-packs/{domain}/prompts.yaml candidate 예시 주입
"""
from __future__ import annotations

from typing import Any

from structverify.detection._llm import get_llm_client
from structverify.detection.prompts.candidate import CANDIDATE_PROMPT
from structverify.detection.prompts_loader import resolve_prompt_for_step
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def _score_candidate_llm(
    sentence: str,
    *,
    config: dict,
    threshold: float,
    domain: str | None = None,
) -> tuple[float, bool, str, dict[str, Any]]:
    llm = get_llm_client(config)
    base = CANDIDATE_PROMPT.format(sentence=sentence)
    prompt = resolve_prompt_for_step(base, domain, config, step="candidate")
    result = await llm.generate_json(
        prompt=prompt,
        system_prompt="팩트체크 candidate detector. JSON으로만 답하세요.",
    )
    # score = float(result.get("candidate_score", 0.0))
    # label = bool(result.get("candidate_label", score >= threshold))
    # signals = result.get("signals", {}) or {}
    # signals["reason"] = result.get("reason")
    # return score, label, "teacher_llm", signals
    score = float(result.get("candidate_score", 0.0) or 0.0)
    label = bool(result.get("candidate_label", score >= threshold))

    # LLM이 label=true인데 score를 0으로 주는 경우 방어
    if label and score < threshold:
        score = max(score, 0.75)

    signals = result.get("signals", {}) or {}
    signals["reason"] = result.get("reason")
    signals["raw_llm_result"] = result

    return score, label, "teacher_llm", signals
