"""
graph/provenance.py — Provenance (출처 이력) 추적

검증 결과가 어떤 커넥터 → 어떤 통계표 → 어떤 수치를 근거로 도출되었는지
전체 경로를 기록하고 렌더링한다.

[참고] Fact Verification on KG via Programmatic Reasoning (EMNLP Findings 2025)
  KG 위에서 검증 경로를 프로그래밍적으로 추적하는 방법론.
"""
from __future__ import annotations
from structverify.core.schemas import (
    ProvenanceRecord, GraphNode, GraphEdge, GraphNodeType, GraphEdgeType)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def build_provenance_subgraph(
    provenance: ProvenanceRecord,
    claim_node_id: str,
    evidence_node_id: str,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """
    Provenance 정보를 그래프 노드/엣지로 변환하여 출처 경로를 기록한다.

    Returns:
        tuple[nodes, edges]: 출처 그래프 구성 요소
    """
    source_node = GraphNode(
        node_id=f"source:{provenance.provenance_id}",
        node_type=GraphNodeType.SOURCE,
        label=provenance.source_connector,
        properties={"source_id": provenance.source_id,
                    "query_used": provenance.query_used},
    )

    edges = [
        GraphEdge(from_node=evidence_node_id, to_node=source_node.node_id,
                  edge_type=GraphEdgeType.SOURCED_FROM),
        GraphEdge(from_node=claim_node_id, to_node=evidence_node_id,
                  edge_type=GraphEdgeType.VERIFIED_BY),
    ]

    return [source_node], edges


def render_provenance_text(provenance: ProvenanceRecord) -> str:
    """Provenance를 사람이 읽을 수 있는 텍스트로 렌더링한다."""
    return (f"출처: {provenance.source_connector} | "
            f"통계표 ID: {provenance.source_id} | "
            f"검색어: {provenance.query_used} | "
            f"조회 시각: {provenance.fetched_at.isoformat()}")
