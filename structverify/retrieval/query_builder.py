"""

retrieval/query_builder.py — 검색 쿼리 생성 (Schema → ConnectorQuery)

[김예슬 - 2026-04-30 / v2]
- build_query()에 raw_claim 추가: LLM Agent가 catalog 검색 시 원문 맥락 활용
- ConnectorQuery.extra_params에 embedding_text 추가:
  pgvector 유사도 검색에 사용할 텍스트 (indicator + population + time_period 조합)

[참고] ProgramFC (Pan et al., NAACL 2023) — https://github.com/mbzuai-nlp/ProgramFC
  structured representation → data source query 변환 구조 참고
"""
from __future__ import annotations
from structverify.core.schemas import Claim
from structverify.retrieval.base_connector import ConnectorQuery


def build_query(claim: Claim) -> ConnectorQuery:
    """
    Claim Schema → 커넥터 검색 쿼리 생성

    [v4 김예슬] extra_params에 추가:
      - raw_claim: 원문 주장 (LLM Agent catalog 검색 시 맥락)
      - embedding_text: pgvector 유사도 검색용 텍스트
    """
    s = claim.schema
    # [v3 김예슬] context_text 있으면 raw_claim에 포함 → 검색 맥락 강화
    context = getattr(claim, "context_text", None) or claim.claim_text

    if not s:
        return ConnectorQuery(
            keyword=claim.claim_text[:50],
            extra_params={
                "raw_claim":      context,
                "embedding_text": context[:200],
            },
        )

    # keyword: source_reference(출처기관) + indicator + population 조합
    # 출처기관이 있으면 KOSIS 검색 범위를 해당 기관으로 좁힐 수 있음
    parts = [p for p in [s.source_reference, s.indicator, s.population] if p]
    keyword = " ".join(parts) or claim.claim_text[:50]

    # [v6.6 변경] embedding_text를 박재윤 factcheck_test.py v6 포맷에 맞춤
    # KOSIS 카탈로그가 인덱싱된 포맷과 매칭되어야 임베딩 거리가 작아짐:
    #   "{categories} > {indicator} | 항목: {indicator} | 분류: {kw} | 단위: {unit}"
    # 단, build_query 시점에는 category_keywords가 없음 (catalog_search 내부 LLM이 추출).
    # 그러므로 여기서는 base_text만 넘기고, catalog_search가 cats를 prepend 함.
    indicator = s.indicator or ""
    unit = s.unit or ""
    population = s.population or ""

    # [v6.8 추가] 단위 정규화 — KOSIS 카탈로그가 인덱싱될 때 사용한 표기에 맞춤
    # 예: KOSIS 데이터에는 "℃"로 인덱싱되어 있는데 우리 claim 단위는 "도"라고 박혀있어서
    # 임베딩 거리가 멀어짐. 의미상 동일한 단위는 KOSIS 표기로 변환.
    # 이건 룰 기반이지만 *단위 명칭 표기* 매핑일 뿐 indicator 매핑 아님.
    _UNIT_NORMALIZE = {
        "도":   "℃",
        "℃":   "℃",
        "도씨": "℃",
        "퍼센트": "%",
        "%":    "%",
        "프로":  "%",
        "위안":  "위안",
        "엔":   "엔",
        "달러":  "달러",
    }
    unit_normalized = _UNIT_NORMALIZE.get(unit, unit)

    embedding_text = (
        f"{indicator} {population} "
        f"| 항목: {indicator} "
        f"| 단위: {unit_normalized}"
    ).strip()

    # [v6.5 기존 — 비교용으로 주석 보존]
    # emb_parts = [p for p in [s.indicator, s.population, s.time_period] if p]
    # embedding_text = " ".join(emb_parts) or keyword

    return ConnectorQuery(
        keyword=keyword,
        indicator=s.indicator,
        time_period=s.time_period,
        population=s.population,
        extra_params={
            "raw_claim":      context,       # [v3] context 포함
            "embedding_text": embedding_text,
            # source_reference가 있으면 KOSIS org_name 필터로 활용
            "source_org":     s.source_reference,
            # [v6.6 추가] unit을 명시적으로 보관 — catalog_search에서 임베딩 텍스트 강화 시 활용
            # [v6.8 변경] 정규화된 단위 저장
            "unit":           unit_normalized,
            "unit_raw":       unit,
        },
    )