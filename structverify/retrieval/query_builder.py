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
    if not s:
        return ConnectorQuery(
            keyword=claim.claim_text[:50],
            extra_params={
                "raw_claim": claim.claim_text,
                "embedding_text": claim.claim_text[:200],
            },
        )

    # keyword: source_reference(출처기관) + indicator + population 조합
    # 출처기관이 있으면 KOSIS 검색 범위를 해당 기관으로 좁힐 수 있음
    parts = [p for p in [s.source_reference, s.indicator, s.population] if p]
    keyword = " ".join(parts) or claim.claim_text[:50]

    # embedding_text: indicator + population + time_period
    # pgvector 유사도 검색에 사용 (source_reference 제외 — 기관명 편향 방지)
    emb_parts = [p for p in [s.indicator, s.population, s.time_period] if p]
    embedding_text = " ".join(emb_parts) or keyword

    return ConnectorQuery(
        keyword=keyword,
        indicator=s.indicator,
        time_period=s.time_period,
        population=s.population,
        extra_params={
            "raw_claim":      claim.claim_text,
            "embedding_text": embedding_text,
            # source_reference가 있으면 KOSIS org_name 필터로 활용
            "source_org":     s.source_reference,
        },
    )