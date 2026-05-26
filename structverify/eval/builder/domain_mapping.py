"""Registry domain → KOSIS category_path filter keywords."""
from __future__ import annotations

DOMAIN_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "agriculture": ["농림", "농업", "농가", "축산", "어업", "수산", "식품"],
    "automotive_technology": ["자동차", "전기차", "모빌리티", "운송"],
    "economy": ["경제", "산업", "물가", "성장", "무역", "gdp", "소비", "생산"],
    "education": ["교육", "학교", "학생", "진학", "대학"],
    "employment": ["고용", "노동", "취업", "임금", "근로", "실업", "일자리"],
    "environment": ["환경", "기상", "기후", "오염", "에너지", "탄소", "대기"],
    "finance": ["금융", "금리", "환율", "주가", "대출", "증권", "보험", "부채"],
    "healthcare": ["보건", "의료", "건강", "질환", "병원", "사망"],
    "policy": ["정책", "행정", "예산", "복지", "법률", "정부", "지원"],
    "population": ["인구", "출생", "사망", "혼인", "이혼", "가구"],
    "real_estate": ["부동산", "주택", "아파트", "건설", "전세", "매매"],
    "weather": ["날씨", "기상", "강수", "기온", "폭염", "한파"],
}


def get_category_keywords(domain: str) -> list[str]:
    return list(DOMAIN_CATEGORY_KEYWORDS.get(domain.lower(), []))


def all_eval_domains(registry_domains: list[str], exclude: list[str] | None = None) -> list[str]:
    exclude_set = {d.lower() for d in (exclude or ["general"])}
    return sorted(d for d in registry_domains if d.lower() not in exclude_set)


def build_category_where_clause(domain: str, param_offset: int = 1) -> tuple[str, list[str]]:
    keywords = get_category_keywords(domain)
    if not keywords:
        return "TRUE", []
    parts = []
    params: list[str] = []
    for i, kw in enumerate(keywords):
        parts.append(f"category_path ILIKE ${param_offset + i}")
        params.append(f"%{kw}%")
    return f"({' OR '.join(parts)})", params
