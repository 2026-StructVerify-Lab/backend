"""
graph/graph_builder.py — Claim/Evidence Graph 조립 (Step 6)

유도된 스키마와 클레임 정보를 바탕으로 그래프 노드/엣지를 생성한다.

[참고] GraphRAG (arXiv 2501.00309)
  텍스트에서 엔티티/관계를 추출하여 그래프를 구성하고 검색에 활용하는 패턴.

[참고] AutoSchemaKG (arXiv 2505.23628) — https://github.com/NousResearch/AutoSchemaKG
  LLM이 유도한 schema_candidates를 그래프 노드/엣지로 변환하는 구조.

[참고] Fact Verification on KG (EMNLP Findings 2025)
  지식 그래프 위에서 프로그래밍적 추론으로 사실을 검증하는 방법론.

[0424(금) 진행 내용 관련 참고 논문] - 이수민
[참고] FEVER (Thorne et al., NAACL 2018) — https://fever.ai/
  Fact Extraction and VERification 벤치마크. Claim을 검증의 최소 단위로 다루고
  SUPPORTS / REFUTES / NOT ENOUGH INFO 3-class verdict 체계를 정립.
  본 모듈의 ClaimNode를 일급 노드로 두는 모델링과 schemas.py의 VerdictType이
  이 체계를 따른다.

[참고] HOVER (Jiang et al., EMNLP 2020) — https://hover-nlp.github.io/
  다중 hop 사실 검증 데이터셋. 한 claim을 판정하려면 여러 증거 노드를 연결한
  서브그래프 추론이 필요함을 실증. Claim을 중심에 두고 Metric/Time/Entity를
  주변 노드로 잇는 본 모듈의 구조적 동기.

[참고] TabFact (Chen et al., ICLR 2020) — https://tabfact.github.io/
  표(통계표) 기반 수치 사실 검증 벤치마크. (indicator, time_period, value)
  형태의 통계 클레임 검증 시나리오에 직접 대응되며, KOSIS 같은 통계표 소스를
  Evidence로 사용하는 본 파이프라인의 근거 모델.

[Step 6 — 담당: 이수민]
  Claim + ClaimSchema
    → 노드 생성:
      - ClaimNode  (claim_id, claim_text)
      - MetricNode (indicator명)
      - TimeNode   (time_period)
      - EntityNode (population/대상)
    → 엣지 생성:
      - claim → MEASURED_AT → time
      - claim → BELONGS_TO  → metric
      - claim → BELONGS_TO  → entity
    → GraphNode[], GraphEdge[] 반환 (Neo4j 저장은 별도)
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

    각 클레임의 schema(ClaimSchema)를 기반으로:
    1) Claim Node 생성 (claim_id, claim_text)
    2) Metric Node 생성 (schema.indicator)
    3) Time Node 생성   (schema.time_period)
    4) Entity Node 생성 (schema.population)
    5) Relation 엣지 생성
        - claim --MEASURED_AT--> time
        - claim --BELONGS_TO --> metric
        - claim --BELONGS_TO --> entity

    Returns:
        tuple[nodes, edges]
    """
    # 전체 그래프 누적 컨테이너
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # 동일 indicator/time_period/population이 여러 claim에 반복 등장할 수 있으므로
    # 노드 중복 생성을 막기 위해 이미 추가된 node_id를 추적한다.
    seen_node_ids: set[str] = set()

    def _add_node(node: GraphNode) -> None:
        """중복 방지하면서 노드를 nodes 리스트에 추가하는 헬퍼"""
        if node.node_id not in seen_node_ids:
            nodes.append(node)
            seen_node_ids.add(node.node_id)

    # ── 클레임 단위로 그래프 조립 ─────────────────────────────────
    for claim in claims:
        # (1) ClaimNode 생성
        # claim_id의 hex 앞 8자리를 사용해 짧고 안정적인 node_id를 만든다.
        # label에는 가독성을 위해 claim_text 앞부분만 잘라 표기한다.
        claim_node_id = f"claim:{claim.claim_id.hex[:8]}"
        _add_node(GraphNode(
            node_id=claim_node_id,
            node_type=GraphNodeType.CLAIM,
            label=claim.claim_text[:60],
            properties={
                # 검증 단계에서 활용할 수 있도록 핵심 속성을 properties에 보관
                "claim_text": claim.claim_text,
                "value": claim.schema.value if claim.schema else None,
                "claim_type": claim.claim_type.value if claim.claim_type else None,
            },
        ))

        # ClaimSchema가 없으면 Metric/Time/Entity를 만들 근거가 없으므로 스킵한다.
        # (이 경우 ClaimNode만 그래프에 남는다.)
        if not claim.schema:
            continue

        schema = claim.schema

        # (2) MetricNode 생성 + claim --BELONGS_TO--> metric
        # indicator(지표명, 예: "실업률")가 있을 때만 노드/엣지를 생성한다.
        if schema.indicator:
            metric_node_id = f"metric:{schema.indicator}"
            _add_node(GraphNode(
                node_id=metric_node_id,
                node_type=GraphNodeType.METRIC,
                label=schema.indicator,
                properties={"unit": schema.unit},  # 단위가 있으면 함께 보관
            ))
            # claim이 어떤 지표에 속하는지를 BELONGS_TO 관계로 표현
            edges.append(GraphEdge(
                from_node=claim_node_id,
                to_node=metric_node_id,
                edge_type=GraphEdgeType.BELONGS_TO,
            ))

        # (3) TimeNode 생성 + claim --MEASURED_AT--> time
        # time_period(예: "2024Q1", "2023년")가 있을 때만 노드/엣지를 생성한다.
        if schema.time_period:
            time_node_id = f"time:{schema.time_period}"
            _add_node(GraphNode(
                node_id=time_node_id,
                node_type=GraphNodeType.TIME,
                label=schema.time_period,
            ))
            # 측정 시점을 표현하는 MEASURED_AT 관계
            edges.append(GraphEdge(
                from_node=claim_node_id,
                to_node=time_node_id,
                edge_type=GraphEdgeType.MEASURED_AT,
            ))

        # (4) EntityNode 생성 + claim --BELONGS_TO--> entity
        # population(대상 집단, 예: "청년층", "전국")이 있을 때만 노드/엣지 생성.
        if schema.population:
            entity_node_id = f"entity:{schema.population}"
            _add_node(GraphNode(
                node_id=entity_node_id,
                node_type=GraphNodeType.ENTITY,
                label=schema.population,
            ))
            # 대상 집단(엔티티)에 대한 소속 관계도 BELONGS_TO로 표현
            edges.append(GraphEdge(
                from_node=claim_node_id,
                to_node=entity_node_id,
                edge_type=GraphEdgeType.BELONGS_TO,
            ))

    logger.info(f"Graph 조립: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


# ──────────────────────────────────────────────────────────────
# [기존 코드: graph_schema_candidates 기반 동적 노드/엣지 생성]
# Step 6 스펙(고정 필드 indicator/time_period/population)을 우선 적용하기 위해
# 아래 LLM 후보 기반 로직은 주석 처리한다. 추후 LLM-유도 스키마 확장이 필요할 때
# 다시 활성화하거나 build_claim_graph와 병행하도록 변경할 수 있다.
# ──────────────────────────────────────────────────────────────
#
# def build_claim_graph(claims: list[Claim]) -> tuple[list[GraphNode], list[GraphEdge]]:
#     """
#     클레임 리스트 → Claim Graph (노드 + 엣지) 조립
#
#     각 클레임의 schema.graph_schema_candidates를 기반으로:
#     1) Claim Node 생성
#     2) Entity/Metric/Time 노드 생성 (LLM이 유도한 후보 기반)
#     3) Relation 엣지 생성
#
#     Returns:
#         tuple[nodes, edges]
#     """
#     nodes: list[GraphNode] = []
#     edges: list[GraphEdge] = []
#
#     for claim in claims:
#         claim_node_id = f"claim:{claim.claim_id.hex[:8]}"
#         nodes.append(GraphNode(
#             node_id=claim_node_id, node_type=GraphNodeType.CLAIM,
#             label=claim.claim_text[:60],
#             properties={"value": claim.schema.value if claim.schema else None,
#                         "claim_type": claim.claim_type.value if claim.claim_type else None},
#         ))
#
#         if not claim.schema or not claim.schema.graph_schema_candidates:
#             # Schema 없으면 기본 노드만 생성
#             _build_default_nodes(claim, claim_node_id, nodes, edges)
#             continue
#
#         # LLM이 유도한 graph_schema_candidates로 노드/엣지 생성
#         for candidate in claim.schema.graph_schema_candidates:
#             if "node_type" in candidate:
#                 node_id = f"{candidate.get('node_type', 'entity')}:{candidate.get('label', 'unknown')}"
#                 ntype = _map_node_type(candidate.get("node_type", "entity"))
#                 nodes.append(GraphNode(
#                     node_id=node_id, node_type=ntype,
#                     label=candidate.get("label", ""),
#                 ))
#             elif "edge_type" in candidate:
#                 etype = _map_edge_type(candidate.get("edge_type", "belongs_to"))
#                 edges.append(GraphEdge(
#                     from_node=f"metric:{candidate.get('from', '')}",
#                     to_node=f"time:{candidate.get('to', '')}",
#                     edge_type=etype,
#                 ))
#
#         # Claim → 각 Entity 연결
#         for n in nodes:
#             if n.node_id != claim_node_id and n.node_type != GraphNodeType.CLAIM:
#                 edges.append(GraphEdge(
#                     from_node=claim_node_id, to_node=n.node_id,
#                     edge_type=GraphEdgeType.BELONGS_TO))
#
#     logger.info(f"Graph 조립: {len(nodes)} nodes, {len(edges)} edges")
#     return nodes, edges


# def _build_default_nodes(claim, claim_node_id, nodes, edges):
#     """Schema 없을 때 기본 노드 생성"""
#     if claim.schema and claim.schema.indicator:
#         mid = f"metric:{claim.schema.indicator}"
#         nodes.append(GraphNode(node_id=mid, node_type=GraphNodeType.METRIC,
#                                label=claim.schema.indicator))
#         edges.append(GraphEdge(from_node=claim_node_id, to_node=mid,
#                                edge_type=GraphEdgeType.BELONGS_TO))


# def _map_node_type(t: str) -> GraphNodeType:
#     mapping = {"entity": GraphNodeType.ENTITY, "metric": GraphNodeType.METRIC,
#                "time": GraphNodeType.TIME, "evidence": GraphNodeType.EVIDENCE}
#     return mapping.get(t, GraphNodeType.ENTITY)


# def _map_edge_type(t: str) -> GraphEdgeType:
#     mapping = {"measured_at": GraphEdgeType.MEASURED_AT,
#                "belongs_to": GraphEdgeType.BELONGS_TO,
#                "sourced_from": GraphEdgeType.SOURCED_FROM}
#     return mapping.get(t, GraphEdgeType.BELONGS_TO)
