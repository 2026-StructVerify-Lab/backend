"""detection/_llm.py — Step 3~5 LLM thin wrapper.

structverify.utils.llm_client.LLMClient를 detection 모듈에서 직접 생성하지 않고
여기를 경유한다. (utils/llm_client.py 대수술 없음)

[리팩 Phase C #14] detection 내 LLMClient 생성은 get_llm_client()로 통일.
[리팩 Phase C #15] llm 서브설정은 detection._config.llm_config() 경유.
"""
from __future__ import annotations

from typing import Any

from structverify.detection._config import llm_config
from structverify.utils.llm_client import LLMClient


def get_llm_client(config: dict | None = None) -> LLMClient:
    """detection._config.llm_config()로 LLMClient 생성 (레거시 config.llm 호환)."""
    return LLMClient(config=llm_config(config))


def llm_config_from(config: dict | None) -> dict[str, Any]:
    """전체 config dict에서 llm 서브 dict 추출 (detection.llm 병합 포함)."""
    return llm_config(config)
