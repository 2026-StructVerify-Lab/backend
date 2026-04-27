"""
retrieval/base_connector.py — 데이터 커넥터 추상 인터페이스

v2: to_graph_nodes(), tag_provenance() 메서드 추가

[참고] RAG (Lewis et al., NeurIPS 2020) — https://github.com/huggingface/transformers
  검색 → LLM context 주입 패턴. 커넥터의 search→fetch→context 흐름에 참고.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from structverify.core.schemas import GraphNode, ProvenanceRecord


@dataclass
class ConnectorQuery:
    keyword: str
    indicator: str | None = None
    time_period: str | None = None
    population: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class StatRecord:
    stat_id: str; stat_name: str; org_name: str; org_id: str | None = None
    available_periods: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    # 출처별 search 응답 한 행 전체; 커넥터·API별 추가 키는 여기로.
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class StatData:
    stat_id: str; stat_name: str
    values: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """
    모든 커넥터의 추상 기본 클래스.
    v2에서 to_graph_nodes()와 tag_provenance() 추가.
    """
    @abstractmethod
    async def search(self, query: ConnectorQuery) -> list[StatRecord]: ...

    @abstractmethod
    async def fetch(self, stat_id: str, params: dict[str, Any]) -> StatData: ...

    @abstractmethod
    def to_graph_nodes(self, data: StatData) -> list[GraphNode]:
        """v2: 조회 결과를 그래프 노드로 변환"""
        ...

    @abstractmethod
    def tag_provenance(self, data: StatData, query: ConnectorQuery) -> ProvenanceRecord:
        """v2: 출처 이력 기록"""
        ...

    async def search_and_fetch(self, query: ConnectorQuery) -> StatData | None:
        records = await self.search(query)
        if not records:
            return None
        best = max(records, key=lambda r: r.relevance_score)
        return await self.fetch(
            best.stat_id,
            {
                "time_period": query.time_period,
                "population": query.population,
                "query": query,
                "stat_record": best,
            },
        )
