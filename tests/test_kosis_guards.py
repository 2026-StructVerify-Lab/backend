import pytest
from structverify.retrieval.kosis_relevance import is_overseas_mismatch, is_specialized_mismatch

@pytest.mark.parametrize("table_name, claim_query, expected", [
    ("합계출산율 - 동북·중앙아시아", "합계출산율 전국", True),
    ("유럽 인구", "인구", True),
    ("유럽 인구 추이", "유럽 인구", False),
    ("시도별 인구", "인구", False),
    ("g20 국가 gdp", "gdp", True),
    ("국제결혼 통계", "결혼", True),
    ("OECD 고용률", "고용률", True),
])
def test_is_overseas_mismatch(table_name, claim_query, expected):
    assert is_overseas_mismatch(table_name, claim_query) is expected

@pytest.mark.parametrize("table_name, claim_query, expected", [
    ("[해양기상] 등표 관측값", "연평균기온", True),
    ("연평균기온", "연평균기온", False),
    ("항공기상 관측", "항공기상 통계", False),
    ("부이 관측 자료", "수온", True),
    ("평균기온", "기온", False),
])
def test_is_specialized_mismatch(table_name, claim_query, expected):
    assert is_specialized_mismatch(table_name, claim_query) is expected
