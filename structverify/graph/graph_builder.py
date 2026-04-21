"""
graph/graph_builder.py — Claim/Evidence Graph 조립 (Step 6)

유도된 스키마와 클레임 정보를 바탕으로 그래프 노드/엣지를 생성한다.

[참고] GraphRAG (arXiv 2501.00309)
  텍스트에서 엔티티/관계를 추출하여 그래프를 구성하고 검색에 활용하는 패턴.

[참고] AutoSchemaKG (arXiv 2505.23628) — https://github.com/NousResearch/AutoSchemaKG
  LLM이 유도한 schema_candidates를 그래프 노드/엣지로 변환하는 구조.

[참고] Fact Verification on KG (EMNLP Findings 2025)
  지식 그래프 위에서 프로그래밍적 추론으로 사실을 검증하는 방법론.
"""
from __future__ import annotations
from uuid import uuid4
from structverify.core.schemas import (
    Claim, GraphNode, GraphEdge, GraphNodeType, GraphEdgeType)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def build_claim_graph(claims: list[Claim]) -> tuple[list[GraphNode], list[GraphEdge]]:
    """
    클레임 리스트 → Claim Graph (노드 + 엣지) 조립

    각 클레임의 schema.graph_schema_candidates를 기반으로:
    1) Claim Node 생성
    2) Entity/Metric/Time 노드 생성 (LLM이 유도한 후보 기반)
    3) Relation 엣지 생성

    Returns:
        tuple[nodes, edges]
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for claim in claims:
        claim_node_id = f"claim:{claim.claim_id.hex[:8]}"
        nodes.append(GraphNode(
            node_id=claim_node_id, node_type=GraphNodeType.CLAIM,
            label=claim.claim_text[:60],
            properties={"value": claim.schema.value if claim.schema else None,
                        "claim_type": claim.claim_type.value if claim.claim_type else None},
        ))

        if not claim.schema or not claim.schema.graph_schema_candidates:
            # Schema 없으면 기본 노드만 생성
            _build_default_nodes(claim, claim_node_id, nodes, edges)
            continue

        # LLM이 유도한 graph_schema_candidates로 노드/엣지 생성
        for candidate in claim.schema.graph_schema_candidates:
            if "node_type" in candidate:
                node_id = f"{candidate.get('node_type', 'entity')}:{candidate.get('label', 'unknown')}"
                ntype = _map_node_type(candidate.get("node_type", "entity"))
                nodes.append(GraphNode(
                    node_id=node_id, node_type=ntype,
                    label=candidate.get("label", ""),
                ))
            elif "edge_type" in candidate:
                etype = _map_edge_type(candidate.get("edge_type", "belongs_to"))
                edges.append(GraphEdge(
                    from_node=f"metric:{candidate.get('from', '')}",
                    to_node=f"time:{candidate.get('to', '')}",
                    edge_type=etype,
                ))

        # Claim → 각 Entity 연결
        for n in nodes:
            if n.node_id != claim_node_id and n.node_type != GraphNodeType.CLAIM:
                edges.append(GraphEdge(
                    from_node=claim_node_id, to_node=n.node_id,
                    edge_type=GraphEdgeType.BELONGS_TO))

    logger.info(f"Graph 조립: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


def _build_default_nodes(claim, claim_node_id, nodes, edges):
    """Schema 없을 때 기본 노드 생성"""
    if claim.schema and claim.schema.indicator:
        mid = f"metric:{claim.schema.indicator}"
        nodes.append(GraphNode(node_id=mid, node_type=GraphNodeType.METRIC,
                               label=claim.schema.indicator))
        edges.append(GraphEdge(from_node=claim_node_id, to_node=mid,
                               edge_type=GraphEdgeType.BELONGS_TO))


def _map_node_type(t: str) -> GraphNodeType:
    mapping = {"entity": GraphNodeType.ENTITY, "metric": GraphNodeType.METRIC,
               "time": GraphNodeType.TIME, "evidence": GraphNodeType.EVIDENCE}
    return mapping.get(t, GraphNodeType.ENTITY)


def _map_edge_type(t: str) -> GraphEdgeType:
    mapping = {"measured_at": GraphEdgeType.MEASURED_AT,
               "belongs_to": GraphEdgeType.BELONGS_TO,
               "sourced_from": GraphEdgeType.SOURCED_FROM}
    return mapping.get(t, GraphEdgeType.BELONGS_TO)
