"""EmbeddingClient 테스트 (#67-A/B). API/DB 없이 순수 + httpx mock."""
import httpx
import pytest

from structverify.utils.embedding_client import EmbeddingClient, EMBEDDING_DIM


# ── 골격 (#67-A) ──

def test_default_provider_is_hcx():
    assert EmbeddingClient().provider == "hcx"


def test_provider_from_config():
    c = EmbeddingClient({"provider": "openai", "model": "text-embedding-3-small",
                         "api_key_env": "FOO"})
    assert c.provider == "openai"
    assert c.model == "text-embedding-3-small"


def test_missing_env_key_returns_empty_not_raise(monkeypatch):
    monkeypatch.delenv("FOO_NONEXISTENT", raising=False)
    c = EmbeddingClient({"provider": "openai", "api_key_env": "FOO_NONEXISTENT"})
    assert c.api_key == ""          # 예외 아님, 빈 문자열로 고정


def test_embedding_dim_constant():
    assert EMBEDDING_DIM == 1024


async def test_unknown_provider_embed_raises_valueerror():
    # gemini 등 미지원 provider → dispatch에서 ValueError
    with pytest.raises(ValueError):
        await EmbeddingClient({"provider": "gemini", "api_key_env": "FOO"}).embed("x")


async def test_unknown_provider_embed_batch_raises_valueerror():
    with pytest.raises(ValueError):
        await EmbeddingClient({"provider": "gemini", "api_key_env": "FOO"}).embed_batch(["a"])


# ── HCX 구현 (#67-B), httpx mock ──

class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _patch_post(monkeypatch, payload, status_code=200):
    async def _fake_post(self, url, **kwargs):  # noqa: ANN001
        return _FakeResp(payload, status_code)
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)


async def test_embed_hcx_returns_vector(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "test-key")
    _patch_post(monkeypatch, {"result": {"embedding": [0.1] * 1024}})
    vec = await EmbeddingClient().embed("안녕")
    assert vec == [0.1] * 1024


async def test_embed_hcx_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("CLOVASTUDIO_API_KEY", raising=False)
    # 키 없으면 네트워크 안 타고 None
    assert await EmbeddingClient().embed("안녕") is None


async def test_embed_hcx_no_result_returns_none(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "test-key")
    _patch_post(monkeypatch, {})  # result 키 없음 → KeyError → 로그 후 None
    assert await EmbeddingClient().embed("안녕") is None


async def test_embed_batch_returns_list(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "test-key")
    _patch_post(monkeypatch, {"result": {"embedding": [0.2] * 1024}})
    res = await EmbeddingClient().embed_batch(["a", "b", "c"])
    assert len(res) == 3
    assert all(v == [0.2] * 1024 for v in res)


# ── OpenAI 호환 (openai/upstage) 구현, httpx 경로 강제 + mock ──

def _patch_openai_httpx(monkeypatch, payload, captured=None, status_code=200):
    # SDK 미설치로 가장해 httpx 폴백 경로 강제 (요청 바디 검증 용이)
    monkeypatch.setattr(
        "structverify.utils.embedding_client._load_async_openai", lambda: None
    )

    async def _fake_post(self, url, **kwargs):  # noqa: ANN001
        if captured is not None:
            captured["url"] = url
            captured["json"] = kwargs.get("json")
        return _FakeResp(payload, status_code)
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)


async def test_embed_openai_returns_vector(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    _patch_openai_httpx(monkeypatch, {"data": [{"embedding": [0.2] * 1024}]})
    vec = await EmbeddingClient({"provider": "openai"}).embed("hi")
    assert vec == [0.2] * 1024


async def test_embed_openai_sends_dimensions_1024(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    cap: dict = {}
    _patch_openai_httpx(monkeypatch, {"data": [{"embedding": [0.2] * 1024}]}, captured=cap)
    await EmbeddingClient({"provider": "openai"}).embed("hi")
    assert cap["json"]["dimensions"] == 1024
    assert cap["url"].endswith("/embeddings")


async def test_embed_upstage_same_path(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", "k")
    cap: dict = {}
    _patch_openai_httpx(monkeypatch, {"data": [{"embedding": [0.3] * 1024}]}, captured=cap)
    vec = await EmbeddingClient({"provider": "upstage"}).embed("hi")
    assert vec == [0.3] * 1024
    assert cap["json"]["dimensions"] == 1024     # 같은 공유 경로 (dimensions 강제)


async def test_embed_openai_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert await EmbeddingClient({"provider": "openai"}).embed("hi") is None
