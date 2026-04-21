"""
graph/graph_store.py — Graph DB 인터페이스 (Neo4j / Memgraph)

Claim Graph, Evidence Graph, Provenance Graph를 저장/조회한다.

[참고] GraphRAG (arXiv 2501.00309)
  Graph 기반 Evidence 검색 — subgraph 조회를 통한 다중 hop 추론 지원
"""
from __future__ import annotations
from structverify.core.schemas import GraphNode, GraphEdge
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class GraphStore:
    """
    Graph DB 추상 인터페이스.
    Neo4j 또는 Memgraph에 노드/엣지를 저장하고 서브그래프를 조회한다.
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "neo4j")
        # TODO: Neo4j Driver 초기화
        # from neo4j import AsyncGraphDatabase
        # self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def save_nodes(self, nodes: list[GraphNode]) -> None:
        """
        노드 리스트를 Graph DB에 저장한다.
        TODO: Cypher MERGE 쿼리로 upsert 구현
        # MERGE (n:{node_type} {node_id: $id}) SET n.label=$label, n.properties=$props
        """
        for n in nodes:
            logger.debug(f"Graph node 저장 stub: {n.node_id}")

    async def save_edges(self, edges: list[GraphEdge]) -> None:
        """
        엣지 리스트를 Graph DB에 저장한다.
        TODO: Cypher MATCH + CREATE 쿼리 구현
        """
        for e in edges:
            logger.debug(f"Graph edge 저장 stub: {e.from_node} → {e.to_node}")

    async def get_subgraph(self, anchor_id: str, hops: int = 2) -> dict:
        """
        특정 앵커 노드에서 N-hop 서브그래프를 조회한다.
        TODO: Cypher MATCH path = (n)-[*1..N]-(m) WHERE n.node_id=$anchor RETURN path
        """
        logger.warning(f"서브그래프 조회 stub: {anchor_id}, {hops} hops")
        return {"nodes": [], "edges": []}

    async def close(self):
        """드라이버 종료"""
        # TODO: self.driver.close()
        pass
