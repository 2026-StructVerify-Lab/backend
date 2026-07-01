"""custom_csv DataSource registry 등록 스모크 테스트.

목적: 본문 구현 전, @register_datasource("custom_csv") 등록 + build_datasource
인스턴스화가 동작하는지만 확인. search_catalog/fetch_evidence는 NotImplementedError
스텁이라 호출하지 않음. DB/API 키 불필요.
"""
from pathlib import Path

import structverify.retrieval.custom_csv_source  # noqa: F401 — @register_datasource 트리거
from structverify.retrieval.custom_csv_source import CustomCSVDataSource
from structverify.retrieval.registry import build_datasource, list_datasources

_FIXTURE = str(Path(__file__).parent / "fixtures" / "sample_custom.csv")


def test_custom_csv_registered():
    assert "custom_csv" in list_datasources()


def test_build_custom_csv_instance():
    ds = build_datasource("custom_csv", {"base_path": "x", "catalog_json": "y"})
    assert isinstance(ds, CustomCSVDataSource)
    assert ds.name == "custom_csv"


async def test_search_catalog_match():
    ds = CustomCSVDataSource(csv_path=_FIXTURE)
    res = await ds.search_catalog("고용률")
    assert res, "고용률 후보가 나와야 함"
    assert all("고용률" in c["name"] for c in res)


async def test_search_catalog_no_match():
    ds = CustomCSVDataSource(csv_path=_FIXTURE)
    res = await ds.search_catalog("없는지표")
    assert res == []


async def test_search_catalog_candidate_keys():
    ds = CustomCSVDataSource(csv_path=_FIXTURE)
    res = await ds.search_catalog("고용률")
    for c in res:
        assert {"id", "name", "score"} <= set(c.keys())


async def test_fetch_evidence_specific_year():
    # fixture에 2024 행이 없어 명세의 "2024"를 실재 연도 "2023"으로 검증 (고용률 전국 2023 = 62.6, %)
    ds = CustomCSVDataSource(csv_path=_FIXTURE)
    ev = await ds.fetch_evidence("고용률|전국", {"time_period": "2023"})
    assert ev is not None
    assert ev["value"] == 62.6
    assert ev["unit"] == "%"
    assert ev["time_period"] == "2023"


async def test_fetch_evidence_latest_when_no_time_period():
    # time_period 없으면 가장 최근 연도(2022 vs 2023 → 2023) 행
    ds = CustomCSVDataSource(csv_path=_FIXTURE)
    ev = await ds.fetch_evidence("고용률|전국")
    assert ev is not None
    assert ev["time_period"] == "2023"
    assert ev["value"] == 62.6


async def test_fetch_evidence_no_match_returns_none():
    ds = CustomCSVDataSource(csv_path=_FIXTURE)
    ev = await ds.fetch_evidence("없는지표|전국", {"time_period": "2023"})
    assert ev is None


async def test_fetch_evidence_keys():
    ds = CustomCSVDataSource(csv_path=_FIXTURE)
    ev = await ds.fetch_evidence("출생아수|전국", {"time_period": "2023"})
    assert ev is not None
    assert {"value", "unit", "time_period"} <= set(ev.keys())
