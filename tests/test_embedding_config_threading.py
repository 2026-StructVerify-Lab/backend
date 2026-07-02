"""config.embedding → CatalogSearchTool 스레딩 테스트 (#67-D A-2).

DoD: config 로 embedding provider 선택이 CatalogSearchTool._embedder까지 도달.
API/DB 없이 provider 속성만 확인(임베딩 호출 X).
"""
import structverify.retrieval.kosis_source  # noqa: F401 — @register_datasource 트리거
from structverify.retrieval.catalog_search import CatalogSearchTool
from structverify.retrieval.registry import build_datasource


def test_catalog_consumes_embedding_config():
    # A-1 소비: config.embedding 있으면 그 provider 사용
    t = CatalogSearchTool({"embedding": {"provider": "openai", "api_key_env": "X"}})
    assert t._embedder.provider == "openai"


def test_catalog_fallback_when_no_embedding():
    # 폴백 보존: embedding 없으면 hcx 기본
    t = CatalogSearchTool({})
    assert t._embedder.provider == "hcx"


def test_runtime_legacy_threads_embedding():
    # A-2 레거시 경로: RuntimeAgent → kosis_cfg → KOSISConnector → CatalogSearchTool
    from structverify.agent.runtime_agent import RuntimeAgent
    rt = RuntimeAgent({"embedding": {"provider": "openai", "api_key_env": "X"}})
    assert rt.kosis.catalog._embedder.provider == "openai"


def test_datasource_passthrough_threads_embedding():
    # 에이전트 경로 패스스루: build_datasource → KOSISDataSource → lazy KOSISConnector → CatalogSearchTool
    ds = build_datasource("kosis", {"embedding": {"provider": "openai", "api_key_env": "X"}})
    conn = ds._get_connector()
    assert conn.catalog._embedder.provider == "openai"
