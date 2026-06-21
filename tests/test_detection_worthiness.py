"""tests/test_detection_worthiness.py — claim worthiness unit tests (API 키 불필요)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from structverify.core.schemas import ClaimType
from structverify.detection.claims.worthiness import _check_worthiness


@pytest.mark.asyncio
async def test_check_worthiness_returns_claim_type_on_positive():
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value={
        "is_check_worthy": True,
        "score": 0.85,
        "claim_type": "scale",
        "canonical_type": "scale",
    })
    score, claim_type, canonical = await _check_worthiness(
        llm,
        "2024년 출생아 수는 2만 171명이다.",
        config={},
        domain=None,
    )
    assert score == 0.85
    assert claim_type == "scale"
    assert canonical == ClaimType.SCALE


@pytest.mark.asyncio
async def test_check_worthiness_not_worthy_returns_zeros(tmp_path):
    pack_dir = tmp_path / "economy"
    pack_dir.mkdir()
    (pack_dir / "prompts.yaml").write_text(
        "claim_worthiness_examples:\n  - 예시\n",
        encoding="utf-8",
    )
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value={
        "is_check_worthy": False,
        "score": 0.1,
    })
    config = {"domain_packs_dir": str(tmp_path)}
    score, claim_type, canonical = await _check_worthiness(
        llm,
        "내일 비가 올 것으로 보인다.",
        config=config,
        domain="economy",
    )
    assert score == 0.0
    assert claim_type is None
    assert canonical is None
    prompt = llm.generate_json.await_args.kwargs.get("prompt") or llm.generate_json.await_args.args[0]
    assert "[도메인 few-shot 예시]" in prompt
