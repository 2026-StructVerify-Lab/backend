"""detection/_llm.py — Step 3~5 LLM thin wrapper.

structverify.utils.llm_client.LLMClient를 detection 모듈에서 직접 생성하지 않고
여기를 경유한다. (utils/llm_client.py 대수술 없음)

[리팩] 초기에는 인스턴스 생성만 중앙화. generate_* 호출은 기존 파일에서
LLMClient를 쓰다가 Step 3~4 정리 + config.detection.* 커밋에서 점진 이전.
"""
from __future__ import annotations

from typing import Any

from structverify.utils.llm_client import LLMClient


def get_llm_client(config: dict | None = None) -> LLMClient:
    """config['llm'] 블록으로 LLMClient 생성 (기존 detection 진입점과 동일)."""
    cfg = config or {}
    return LLMClient(config=cfg.get("llm", {}))


def llm_config_from(config: dict | None) -> dict[str, Any]:
    """전체 config dict에서 llm 서브 dict만 추출."""
    cfg = config or {}
    return cfg.get("llm", {}) or {}
