"""CatalogSearchTool._get_embedding 특성화 테스트 (#67-D A-1).

교체(인라인 HCX → EmbeddingClient.embed) 전후 동작 고정. httpx post mock(key-aware).
계약: 성공 → list[float] / 키 없음·result 없음 → None.
반환은 _search_pgvector 입력형(float 리스트, str(v) join)으로 보존돼야 함.
"""
import httpx

from structverify.retrieval.catalog_search import CatalogSearchTool


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


async def test_get_embedding_returns_list(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    vec = await CatalogSearchTool({})._get_embedding("hi")
    assert isinstance(vec, list)
    assert vec == [0.1] * 1024


async def test_get_embedding_none_on_no_key(monkeypatch):
    monkeypatch.delenv("CLOVASTUDIO_API_KEY", raising=False)
    monkeypatch.delenv("NCP_API_KEY", raising=False)
    _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    assert await CatalogSearchTool({})._get_embedding("hi") is None


async def test_get_embedding_none_on_no_result(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    _patch(monkeypatch, keyed_payload={})  # result 없음
    assert await CatalogSearchTool({})._get_embedding("hi") is None


async def test_get_embedding_shape_for_pgvector(monkeypatch):
    # _search_pgvector:498 와 동일하게 'for v in embedding' + str(v) join 가능해야 함
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.5, 0.25]}})
    vec = await CatalogSearchTool({})._get_embedding("hi")
    assert vec == [0.5, 0.25]
    vstr = "[" + ",".join(str(v) for v in vec) + "]"
    assert vstr == "[0.5,0.25]"
