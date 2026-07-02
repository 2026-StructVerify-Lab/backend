"""kosis_crawler.save_to_db 임베딩 경로 특성화 테스트 (#67-D B).

교체(인라인 get_embedding_safe → EmbeddingClient.embed_batch) 전후 동작 고정.
psycopg2(import) + httpx(post) 를 mock 해 DB/네트워크 없이 배치 임베딩만 검증.

mock post는 Authorization 헤더의 키 유무로 응답을 가른다(실 HCX API가 무키면 result를
안 주는 상황 모사) → 교체 전(키 빈 채로 POST)과 교체 후(EmbeddingClient가 무키 시
short-circuit)가 모두 zero 폴백으로 수렴.
"""
import sys
import types

import httpx

from structverify.adaptation import kosis_crawler


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def close(self):
        pass


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        pass

    def close(self):
        pass


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _patch(monkeypatch, *, keyed_payload, keyless_payload=None, status_code=200):
    if keyless_payload is None:
        keyless_payload = {}
    fake_conn = _FakeConn()

    # save_to_db 내부 `import psycopg2` 가 fake 모듈을 받게 함 (psycopg2 미설치여도 동작)
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = lambda **kw: fake_conn
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    async def _fake_post(self, url, **kwargs):  # noqa: ANN001
        auth = (kwargs.get("headers") or {}).get("Authorization", "")
        key = auth.replace("Bearer", "").strip()
        return _Resp(keyed_payload if key else keyless_payload, status_code)
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    return fake_conn


def _emb_param(call):
    # execute params 순서: (stat_id, stat_name, org_id, org_name, category_path,
    #                        keywords, embedding, raw_meta_json) → embedding = index 6
    return call[1][6]


def _item(i=0, **over):
    base = {
        "stat_id": f"T{i}", "stat_name": f"n{i}", "org_id": "O",
        "org_name": "통계청", "category_path": "c", "keywords": [],
    }
    base.update(over)
    return base


async def test_save_to_db_embeds_and_inserts(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    conn = _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    n = await kosis_crawler.save_to_db([_item(0, stat_name="고용률")], {})
    assert n == 1
    calls = conn.cursor_obj.calls
    assert len(calls) == 1
    assert _emb_param(calls[0]) == str([0.1] * 1024)


async def test_save_to_db_zero_fallback_when_no_result(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    conn = _patch(monkeypatch, keyed_payload={})  # result 없음 → zero 폴백
    await kosis_crawler.save_to_db([_item(0)], {})
    assert _emb_param(conn.cursor_obj.calls[0]) == str([0.0] * 1024)


async def test_save_to_db_zero_fallback_when_no_key(monkeypatch):
    monkeypatch.delenv("CLOVASTUDIO_API_KEY", raising=False)
    monkeypatch.delenv("NCP_API_KEY", raising=False)
    conn = _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.1] * 1024}})
    await kosis_crawler.save_to_db([_item(0)], {})
    assert _emb_param(conn.cursor_obj.calls[0]) == str([0.0] * 1024)


async def test_save_to_db_batch_length(monkeypatch):
    monkeypatch.setenv("CLOVASTUDIO_API_KEY", "k")
    conn = _patch(monkeypatch, keyed_payload={"result": {"embedding": [0.2] * 1024}})
    catalog = [_item(i) for i in range(3)]
    n = await kosis_crawler.save_to_db(catalog, {})
    assert n == 3
    assert len(conn.cursor_obj.calls) == 3
    assert all(_emb_param(c) == str([0.2] * 1024) for c in conn.cursor_obj.calls)
