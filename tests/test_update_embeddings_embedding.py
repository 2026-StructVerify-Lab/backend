"""update_embeddings.process_single_table 임베딩 경로 특성화 테스트 (#67-D D).

교체(인라인 HCX → EmbeddingClient.embed) 전후 동작 고정.
httpx get(KOSIS 메타)·post(임베딩) 전역 mock. process_single_table을 직접 호출(4-arg)해
교체 전(인라인 client.post) / 후(embedder=None→내부 EmbeddingClient) 모두 같은 mock으로 검증.

핵심 계약: 임베딩 성공 → (embedding, stat_id) 튜플 / 실패(키·result 없음) → None(skip).
"""
import asyncio
import sys
import types

import httpx

# update_embeddings.py가 top-level로 psycopg2를 import 하는데 이 env엔 미설치.
# process_single_table은 psycopg2를 안 쓰므로(메인만 DB) import만 통과시키는 shim.
if "psycopg2" not in sys.modules:
    _pg = types.ModuleType("psycopg2")
    _pg.connect = lambda **kw: None
    _pg_extras = types.ModuleType("psycopg2.extras")
    _pg_extras.execute_values = lambda *a, **k: None
    _pg.extras = _pg_extras
    sys.modules["psycopg2"] = _pg
    sys.modules["psycopg2.extras"] = _pg_extras

from structverify.adaptation import update_embeddings as ue


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _patch_http(monkeypatch, *, keyed_payload, keyless_payload=None):
    if keyless_payload is None:
        keyless_payload = {}

    async def _fake_get(self, url, **kwargs):  # noqa: ANN001
        # KOSIS 딥메타 → err 31 → meta=None (embed_text는 기본 이름)
        return _Resp({"err": "31"})

    async def _fake_post(self, url, **kwargs):  # noqa: ANN001
        auth = (kwargs.get("headers") or {}).get("Authorization", "")
        key = auth.replace("Bearer", "").strip()
        return _Resp(keyed_payload if key else keyless_payload)

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)


def _item(i=0):
    return {"stat_id": f"T{i}", "stat_name": f"n{i}", "org_id": "O", "category_path": "c"}


async def _run(item):
    sem = asyncio.Semaphore(1)
    async with httpx.AsyncClient() as client:
        return await ue.process_single_table(client, item, sem, sem)


async def test_returns_tuple_on_success(monkeypatch):
    monkeypatch.setattr(ue, "HCX_API_KEY", "k", raising=False)
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    _patch_http(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    res = await _run(_item(0))
    assert res == ([0.1] * 1024, "T0")


async def test_skip_none_on_no_result(monkeypatch):
    monkeypatch.setattr(ue, "HCX_API_KEY", "k", raising=False)
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    _patch_http(monkeypatch, keyed_payload={})  # result 없음 → skip(None)
    res = await _run(_item(0))
    assert res is None


async def test_skip_none_on_no_key(monkeypatch):
    monkeypatch.setattr(ue, "HCX_API_KEY", "", raising=False)
    monkeypatch.delenv("CLOVASTUDIO_API_KEY", raising=False)
    monkeypatch.delenv("NCP_API_KEY", raising=False)
    _patch_http(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    res = await _run(_item(0))
    assert res is None
