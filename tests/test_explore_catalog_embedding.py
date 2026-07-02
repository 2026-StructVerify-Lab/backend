"""explore_catalog._embed_query 임베딩 경로 특성화 테스트 (#67-D C).

교체(인라인 HCX → EmbeddingClient.embed) 전후 동작 고정. httpx post mock(key-aware).
계약: 임베딩 성공 → 벡터 / 키 없음·result 없음 → None.
키 우선순위는 NCP_API_KEY 1순위(교체 전: NCP or CLOVA, 교체 후: NCP 단일).
"""
import httpx

from structverify.agent.tools.explore_catalog import ExploreCatalogTool


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _patch(monkeypatch, *, keyed_payload, keyless_payload=None):
    if keyless_payload is None:
        keyless_payload = {}

    async def _fake_post(self, url, **kwargs):  # noqa: ANN001
        auth = (kwargs.get("headers") or {}).get("Authorization", "")
        key = auth.replace("Bearer", "").strip()
        return _Resp(keyed_payload if key else keyless_payload)
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)


async def test_embed_query_returns_vector(monkeypatch):
    monkeypatch.setenv("NCP_API_KEY", "k")
    _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    vec = await ExploreCatalogTool()._embed_query("hi")
    assert vec == [0.1] * 1024


async def test_embed_query_none_on_no_key(monkeypatch):
    monkeypatch.delenv("NCP_API_KEY", raising=False)
    monkeypatch.delenv("CLOVASTUDIO_API_KEY", raising=False)
    _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    assert await ExploreCatalogTool()._embed_query("hi") is None


async def test_embed_query_none_on_no_result(monkeypatch):
    monkeypatch.setenv("NCP_API_KEY", "k")
    _patch(monkeypatch, keyed_payload={})  # result 없음
    assert await ExploreCatalogTool()._embed_query("hi") is None
