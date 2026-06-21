"""tests/test_detection_llm.py — detection._llm wrapper unit tests (API 키 불필요)."""
from structverify.detection._llm import get_llm_client, llm_config_from
from structverify.utils.llm_client import LLMClient


def test_get_llm_client_returns_llm_client():
    client = get_llm_client({"llm": {"provider": "hcx", "temperature": 0.2}})
    assert isinstance(client, LLMClient)
    assert client.provider == "hcx"
    assert client.temperature == 0.2


def test_get_llm_client_none_llm_block_uses_empty_config():
    client = get_llm_client({"llm": None})
    assert isinstance(client, LLMClient)
    assert client.config == {}


def test_llm_config_from_extracts_subdict():
    assert llm_config_from({"llm": {"provider": "openai"}}) == {"provider": "openai"}
    assert llm_config_from(None) == {}
