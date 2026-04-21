"""
retrieval/query_builder.py — 검색 쿼리 생성 (Schema → ConnectorQuery)

[참고] ProgramFC (Pan et al., NAACL 2023) — https://github.com/mbzuai-nlp/ProgramFC
  structured representation → data source query 변환 구조 참고
"""
from __future__ import annotations
from structverify.core.schemas import Claim
from structverify.retrieval.base_connector import ConnectorQuery


def build_query(claim: Claim) -> ConnectorQuery:
    """Claim Schema → 커넥터 검색 쿼리 생성"""
    s = claim.schema
    if not s:
        return ConnectorQuery(keyword=claim.claim_text[:50])
    parts = [p for p in [s.source_reference, s.indicator, s.population] if p]
    return ConnectorQuery(keyword=" ".join(parts) or claim.claim_text[:50],
                          indicator=s.indicator, time_period=s.time_period,
                          population=s.population)
