# tests/test_is_table_relevant.py
"""
_is_table_relevant 특성화(characterization) 테스트.
목적: W2 리팩토링(kosis_source의 함수 import 결합 제거) 전에 현재 동작을 고정.
이 테스트가 계속 초록 = 동작 안 바뀜 증명. 순수 함수라 DB/API 키 불필요.
"""
import pytest
from structverify.retrieval.kosis_relevance import is_table_relevant


@pytest.mark.parametrize("indicator, table_name, expected", [
    # 엣지: 빈 입력이면 필터 안 함(통과)
    ("", "한국 인구", True),
    ("인구", "", True),
    # 1단계 차단 규칙 (False = 버림)
    ("미국 실업률", "시도별 실업률", False),    # 외국 지표 + 국내 표
    ("실업률", "OECD 회원국 실업률", False),    # 국제기구 표
    ("인구", "장래인구추계", False),            # 추계/전망 표
    ("인구", "아프리카 인구 현황", False),       # 해외 지역 표
    ("사망자 수", "영아 사망률", False),         # 세부대상 불일치
    # 2단계 양성 매칭 (True = 통과)
    ("고용률", "청년 고용률 통계", True),        # 직접 토큰 겹침
    ("폭염 일수", "평균기온 변화", True),        # 같은 도메인 그룹(기상)
    # 매칭 근거 없음 (False)
    ("고용률", "아파트 분양 현황", False),
])
def test_is_table_relevant(indicator, table_name, expected):
    assert is_table_relevant(indicator, table_name) is expected
