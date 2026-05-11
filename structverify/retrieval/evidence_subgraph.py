"""
retrieval/evidence_subgraph.py — Evidence 서브그래프 조립 (Step 7)

EvidencePlan 지원 — claim의 value_role에 따라 1~2개 evidence를 가져옴.
첫 번째는 catalog 검색으로 통계표 확정, 두 번째는 같은 표에서 시점만 바꿔 fetch.
이래야 endpoint_a/b의 indicator·단위·지역 정의가 일치 → 의미있는 비교.

[참고] GraphRAG (arXiv 2501.00309)

[참고] GraphRAG (arXiv 2501.00309)
  Evidence를 그래프 구조로 조립하여 다중 hop 추론을 지원하는 패턴
"""
# 수정자: 신준수
# 수정 날짜: 2026-04-27
# 수정 내용: Evidence 필드는 StatData 정규화 필드만 사용( values 키 직접 접근 제거)

from __future__ import annotations

from structverify.core.schemas import (
    Evidence, EvidencePlan, EvidenceRequirement,
    GraphNode, GraphEdge, GraphEdgeType,
)
from structverify.retrieval.base_connector import StatData, ConnectorQuery
from structverify.retrieval.kosis_connector import KOSISConnector
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def build_evidence_subgraph(
    connector: KOSISConnector,
    query: ConnectorQuery,
    claim_node_id: str,
    memory: "DocumentMemory | None" = None,   # [v6.4 추가] 옵셔널 — 기존 호출자 호환
) -> tuple[Evidence | None, list[GraphNode], list[GraphEdge]]:
    """
    [v6.2 호환] 단일 evidence 조회 — measurement 케이스 호환 유지.
    [v6.4 추가] memory 인자 옵셔널 — 캐시 + indicator 재사용.
    """
    data = await connector.search_and_fetch(query, memory=memory)  # [v6.4 변경] memory 전달
    if data is None:
        logger.warning("Evidence 데이터 없음")
        return None, [], []

    evidence = _stat_data_to_evidence(connector, data, role="primary", label="primary")
    nodes, edges = _build_subgraph_for_evidences([evidence], claim_node_id, connector)
    logger.info(f"Evidence 서브그래프: {len(nodes)} nodes")
    return evidence, nodes, edges


async def build_evidence_subgraph_for_plan(
    connector: KOSISConnector,
    base_query: ConnectorQuery,
    plan: EvidencePlan,
    claim_node_id: str,
    memory: "DocumentMemory | None" = None,   # [v6.4 추가] 옵셔널 — 기존 호출자 호환
) -> tuple[list[Evidence], list[GraphNode], list[GraphEdge]]:
    """
    [v6.3 신규] EvidencePlan 기반 multi-evidence 조회.
    [v6.4 추가] memory — 같은 표/시점 재조회 방지.

    전략:
      1) endpoint_a (또는 첫 항목)을 anchor로 잡고 catalog → agent → fetch
      2) 성공 시 그 통계표를 lock (StatData.stat_record 보관)
      3) 나머지 requirements는 fetch_with_locked_table로 같은 표에서 시점만 변경
      4) 모든 evidence를 list로 반환 (combiner가 결합)

    이렇게 하면 endpoint_a와 endpoint_b가 *반드시* 같은 통계표 → 의미 동일성 보장.
    """
    if not plan.requirements:
        logger.info("evidence_plan.requirements 비어있음 — KOSIS 조회 skip")
        return [], [], []

    anchor_req = next(
        (r for r in plan.requirements if r.role == "endpoint_a"),
        None,
    ) or plan.requirements[0]

    anchor_query = ConnectorQuery(
        keyword=anchor_req.indicator or base_query.keyword,
        indicator=anchor_req.indicator or base_query.indicator,
        time_period=anchor_req.time_period or base_query.time_period,
        population=anchor_req.population or base_query.population,
        extra_params=dict(base_query.extra_params),
    )

    logger.info(
        f"[plan] anchor 조회: role={anchor_req.role} "
        f"time={anchor_query.time_period} indicator={anchor_query.indicator}"
    )

    anchor_data = await connector.search_and_fetch(anchor_query, memory=memory)  # [v6.4 변경]
    if anchor_data is None:
        logger.warning("[plan] anchor evidence 없음 → 전체 plan 포기")
        return [], [], []

    anchor_evidence = _stat_data_to_evidence(
        connector, anchor_data,
        role=anchor_req.role,
        label=anchor_req.label or anchor_req.role,
    )
    evidences: list[Evidence] = [anchor_evidence]

    # 나머지 항목: 같은 통계표에서 시점만 바꿔 fetch
    for req in plan.requirements:
        if req is anchor_req:
            continue
        if not req.time_period:
            logger.debug(f"[plan] {req.role}: time_period 없음 → skip")
            continue

        logger.info(
            f"[plan] locked 조회: role={req.role} time={req.time_period} "
            f"(같은 표 [{anchor_data.stat_id}])"
        )
        locked_data = await connector.fetch_with_locked_table(
            stat_record=anchor_data.stat_record,
            time_period=req.time_period,
            original_query=anchor_query,
            memory=memory,   # [v6.4 추가]
        )
        if locked_data is None:
            logger.warning(f"[plan] {req.role} locked-fetch 실패 → skip")
            continue

        ev = _stat_data_to_evidence(
            connector, locked_data,
            role=req.role,
            label=req.label or req.role,
        )
        evidences.append(ev)

    nodes, edges = _build_subgraph_for_evidences(evidences, claim_node_id, connector)
    logger.info(
        f"[plan] Evidence 서브그래프: {len(evidences)}개 evidence, "
        f"{len(nodes)} nodes"
    )
    return evidences, nodes, edges


# ── helpers ──────────────────────────────────────────────────────────────

def _stat_data_to_evidence(
    connector: KOSISConnector,
    data: StatData,
    role: str,
    label: str,
) -> Evidence:
    """StatData → Evidence 변환 + role/label 부착."""
    graph_nodes = connector.to_graph_nodes(data)
    suffix = (data.time_period or role).replace(" ", "")
    for gn in graph_nodes:
        gn.node_id = f"{gn.node_id}:{suffix}"

    provenance = connector.tag_provenance(
        data, ConnectorQuery(keyword=data.stat_name)
    )
    return Evidence(
        source_name=data.stat_name,
        stat_table_id=data.stat_id,
        official_value=data.official_value,
        unit=data.unit,
        time_period=data.time_period,
        raw_response=data.raw_response,
        graph_nodes=graph_nodes,
        provenance=provenance,
        requirement_role=role,
        requirement_label=label,
    )


def _build_subgraph_for_evidences(
    evidences: list[Evidence],
    claim_node_id: str,
    connector: KOSISConnector,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """모든 evidence의 노드를 모으고 claim과 VERIFIED_BY 엣지 연결."""
    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    for ev in evidences:
        for gn in ev.graph_nodes:
            all_nodes.append(gn)
            all_edges.append(GraphEdge(
                from_node=claim_node_id,
                to_node=gn.node_id,
                edge_type=GraphEdgeType.VERIFIED_BY,
                properties={
                    "role": ev.requirement_role or "primary",
                    "label": ev.requirement_label or "",
                },
            ))
    return all_nodes, all_edges