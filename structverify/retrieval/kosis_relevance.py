import re


def is_table_relevant(indicator: str, table_name: str) -> bool:
    """
    [v5] indicator와 테이블명이 같은 분야인지 키워드 검사.
    LLM 미사용 — 키워드 겹침 + 도메인 그룹 매칭 (빠름).

    "연평균기온" vs "기술개발 성과"      → False
    "연평균기온" vs "종관기상 평년값"    → True
    "평균 최저기온" vs "고온 일수 노출"  → True (기온 그룹)
    "시가총액" vs "산업별 취업자"        → False

    [박재윤 - 2026-05-14]: 국가 불일치 체크 추가
    indicator에 외국 국가명 있는데 테이블이 국내 통계면 → False

    [박재윤 - 2026-05-18]: 세부 대상 불일치 차단
    전체 사망자 indicator인데 영아사망 테이블 매칭되는 문제 방지

    [박재윤 - 2026-05-18]: UN/IMF 국제 데이터 테이블 차단
    """
    import re

    if not indicator or not table_name:
        return True

    ind_lower = indicator.lower()
    tbl_lower = table_name.lower()

    # [박재윤 - 2026-05-14]: 국가 불일치 체크
    _FOREIGN_COUNTRIES = {"미국", "일본", "중국", "독일", "영국", "유럽", "프랑스", "캐나다"}
    _FOREIGN_TABLE_MARKERS = {"국제", "해외", "외국", "세계"}
    foreign_in_indicator = any(c in ind_lower for c in _FOREIGN_COUNTRIES)
    foreign_in_table = any(c in tbl_lower for c in _FOREIGN_COUNTRIES | _FOREIGN_TABLE_MARKERS)
    if foreign_in_indicator and not foreign_in_table:
        return False

    # [박재윤 - 2026-05-18]: UN/IMF 국제 데이터 테이블 차단
    _INTERNATIONAL_MARKERS = {"un ", "imf ", "oecd ", "세계은행", "국제연합"}
    if any(m in tbl_lower for m in _INTERNATIONAL_MARKERS):
        return False

    # [박재윤 - 2026-05-14]: 장래 추계 테이블 차단
    _FORECAST_TABLE_MARKERS = {"장래", "추계", "전망"}
    if any(r in tbl_lower for r in _FORECAST_TABLE_MARKERS):
        return False

    # [박재윤 - 2026-05-14]: 해외 지역명 체크 추가
    _REGION_MARKERS = {"남부·동남아시아", "동남아시아", "동북아시아", "중앙아시아",
                       "아프리카", "중동", "남미", "북미", "오세아니아", "서남아시아"}
    region_in_table = any(r in tbl_lower for r in _REGION_MARKERS)
    if not foreign_in_indicator and region_in_table:
        return False

    # [박재윤 - 2026-05-18]: 세부 대상 불일치 차단
    # 전체 사망자/출생아 indicator인데 영아·신생아 테이블 매칭되는 문제
    _SPECIFIC_SCOPE_MARKERS = {"영아", "신생아", "모성"}
    table_is_specific = any(m in tbl_lower for m in _SPECIFIC_SCOPE_MARKERS)
    if table_is_specific:
        ind_has_specific = any(m in ind_lower for m in _SPECIFIC_SCOPE_MARKERS)
        if not ind_has_specific:
            return False

    stopwords = {
        "통계", "현황", "조사", "결과", "항목", "지표", "전체", "기타",
        "관련", "변화", "추이", "비교", "분류", "지점별", "행정구역별",
        "성별", "연령별", "시도별", "시도", "연도별", "월별",
    }

    def _tokens(text: str) -> set[str]:
        words = set(re.split(r'[\s·,/()（）\-_]+', text.lower()))
        return {w for w in words if len(w) >= 2 and w not in stopwords}

    ind_tokens = _tokens(indicator)
    tbl_tokens = _tokens(table_name)

    # 1) 직접 키워드 겹침 (2글자 이상 토큰)
    overlap = ind_tokens & tbl_tokens
    if overlap:
        return True

    # 2) 부분 문자열 포함 (indicator 토큰이 테이블명에 포함)
    tbl_lower = table_name.lower()
    for tok in ind_tokens:
        if len(tok) >= 3 and tok in tbl_lower:
            return True

    # 3) 도메인 키워드 그룹 — 같은 그룹이면 관련 있음
    _GROUPS = [
        # 기상/기후
        {"기온", "기상", "기후", "날씨", "온도", "폭염", "한파", "평균기온",
         "연평균", "최저기온", "최고기온", "강수", "강우", "습도", "종관"},
        # 인구/가구
        {"인구", "출산", "사망", "고령", "청년", "세대", "가구", "출생", "혼인"},
        # 고용/노동
        {"고용", "실업", "취업", "임금", "근로", "노동", "쉬었음", "경제활동",
         "일자리", "실업률", "고용률"},
        # 경제/산업
        {"경제", "성장", "물가", "소비", "생산", "수출", "수입", "무역", "gdp"},
        # 주식/금융
        {"주가", "시총", "코스피", "코스닥", "시가총액", "금리", "환율", "주식"},
        # 교육
        {"교육", "학교", "학생", "진학", "졸업", "입학"},
        # 보건/의료
        {"의료", "건강", "질환", "사망률", "병원", "보건"},
        # 환경
        {"환경", "오염", "대기", "수질", "폐기물", "탄소", "에너지"},
        # 농업
        {"농업", "농가", "농림", "수확", "경작", "과수"},
        # 부동산/건설
        {"주택", "건설", "부동산", "아파트", "토지", "분양"},
    ]

    ind_lower = indicator.lower()
    for group in _GROUPS:
        ind_hit = any(kw in ind_lower for kw in group)
        tbl_hit = any(kw in tbl_lower for kw in group)
        if ind_hit and tbl_hit:
            return True

    return False
