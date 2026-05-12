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

    #   "카테고리 > indicator | 항목: indicator | 분류: population | 단위: unit"
    # KOSIS catalog의 임베딩 인덱싱 포맷과 정렬되어 유사도 검색 정확도 향상.
    emb_cat = s.parent_path or ""
    emb_indicator = s.indicator or ""
    emb_population = s.population or ""
    emb_unit = s.unit or ""
    if emb_indicator:
        embedding_text = (
            f"{emb_cat} > {emb_indicator} | "
            f"항목: {emb_indicator} | "
            f"분류: {emb_population} | "
            f"단위: {emb_unit}"
        ).strip()
    else:
        embedding_text = keyword

    return ConnectorQuery(
        keyword=keyword,
        indicator=s.indicator,
        time_period=s.time_period,
        population=s.population,
        extra_params={
            "raw_claim":      context,
            "embedding_text": embedding_text,
            "source_org":     s.source_reference,
            # [v6.11] parent_path 전달 → catalog_search가 LLM 호출 없이 활용
            "parent_path":    s.parent_path,
        },
    )