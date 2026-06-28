"""RuntimeAgent._build_datasources 배선 테스트 (#66).

config.data_sources.enabled 기반으로 DataSource dict가 구성되는지 + custom_csv
registry 등록(import 트리거)이 동작하는지. API/DB 없이 인스턴스 구성만 확인.
"""
from structverify.agent.runtime_agent import RuntimeAgent
from structverify.retrieval.custom_csv_source import CustomCSVDataSource
from structverify.retrieval.registry import list_datasources


def test_default_enabled_is_kosis():
    # config 없으면 enabled 기본 ["kosis"] → kosis 들어감 (동작 보존)
    ds = RuntimeAgent({})._build_datasources()
    assert "kosis" in ds


def test_config_enables_custom_csv():
    rt = RuntimeAgent({
        "data_sources": {
            "enabled": ["kosis", "custom_csv"],
            "custom_csv": {"csv_path": "x"},
        }
    })
    ds = rt._build_datasources()
    assert {"kosis", "custom_csv"} <= set(ds)
    assert isinstance(ds["custom_csv"], CustomCSVDataSource)


def test_custom_csv_registered_via_trigger():
    # _build_datasources가 custom_csv_source를 import → registry 등록
    RuntimeAgent({})._build_datasources()
    assert "custom_csv" in list_datasources()


def test_custom_csv_config_threads_to_instance():
    # data_sources.custom_csv 의 키가 CustomCSVDataSource(**config)로 도달
    rt = RuntimeAgent({
        "data_sources": {
            "enabled": ["custom_csv"],
            "custom_csv": {"csv_path": "/tmp/foo.csv", "base_path": "/data"},
        }
    })
    ds = rt._build_datasources()
    csv = ds["custom_csv"]
    assert csv.csv_path == "/tmp/foo.csv"
    assert csv.base_path == "/data"
