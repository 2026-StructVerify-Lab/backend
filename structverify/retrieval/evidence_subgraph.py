"""
retrieval/evidence_subgraph.py — Evidence 서브그래프 조립 (Step 7)

커넥터 결과를 Evidence 서브그래프로 변환하여 Claim Graph에 연결한다.

[참고] GraphRAG (arXiv 2501.00309)
  Evidence를 그래프 구조로 조립하여 다중 hop 추론을 지원하는 패턴
"""
from __future__ import annotations
from structverify.core.schemas import (
    Evidence, GraphNode, GraphEdge, GraphNodeType, GraphEdgeType, ProvenanceRecord)
from structverify.retrieval.base_connector import StatData, ConnectorQuery
from structverify.retrieval.kosis_connector import KOSISConnector
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def build_evidence_subgraph(
    connector: KOSISConnector,
    query: ConnectorQuery,
    claim_node_id: str,
) -> tuple[Evidence | None, list[GraphNode], list[GraphEdge]]:
    """
    커넥터로 데이터 조회 → Evidence 객체 + 서브그래프 생성

    Returns:
        (Evidence, [GraphNode], [GraphEdge]) 또는 (None, [], [])
    """
    data = await connector.search_and_fetch(query)
    if data is None:
        logger.warning("Evidence 데이터 없음")
        return None, [], []

    # Evidence 객체 생성
    graph_nodes = connector.to_graph_nodes(data)
    provenance = connector.tag_provenance(data, query)

    evidence = Evidence(
        source_name=data.stat_name, stat_table_id=data.stat_id,
        official_value=data.values.get("ratio") or data.values.get("value"),
        raw_response=data.raw_response,
        graph_nodes=graph_nodes, provenance=provenance,
    )

    # Evidence → Claim 연결 엣지
    edges = []
    for gn in graph_nodes:
        edges.append(GraphEdge(
            from_node=claim_node_id, to_node=gn.node_id,
            edge_type=GraphEdgeType.VERIFIED_BY))

    logger.info(f"Evidence 서브그래프: {len(graph_nodes)} nodes")
    return evidence, graph_nodes, edges
