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
  
  [pipeline 수정 - 담당 : 김예슬]
  - build_claim_graph() 시그니처 변경: sir_doc 파라미터 추가
  · sir_doc이 있으면 extract_context_edges() 호출 → NEXT_SENT/IN_BLOCK/IN_DOC 추가
  · GraphRAG 2-hop 탐색의 핵심 — 같은 문단 내 연관 수치 자동 발견
 
- COMPARE 엣지 추가 (핵심 변경):
  · 같은 MetricNode(indicator)를 공유하는 Claim 쌍에 COMPARE 엣지 생성
  · 예: C1("21만7천") ↔ C2("8만4천") → 둘 다 indicator="쉬었음인구"
  · → C3("2.6배") 같은 파생 주장 검증 시 C1+C2를 함께 KOSIS 조회 가능
  · 같은 지표 여러 시점: C7(12.77개월/2024) ↔ C8(10.71개월/2004)
    → "첫취업소요기간이 2.06개월 늘었다"는 주장도 C7+C8 동시 검증
 
- 문맥 엣지를 GraphEdge 객체로 변환하여 반환값에 포함
  · graph_store.py(박재윤)에서 Neo4j MERGE 시 일괄 처리 가능
 
"""
from __future__ import annotations
 
from collections import defaultdict
from typing import Any
 
from structverify.core.schemas import (
    Claim, GraphEdge, GraphEdgeType, GraphNode, GraphNodeType, SIRDocument,ClaimType
)
from structverify.utils.logger import get_logger
 
logger = get_logger(__name__)
 
# 문맥 엣지 타입 문자열 → GraphEdgeType 매핑
_CONTEXT_EDGE_MAP = {
    "NEXT_SENT": GraphEdgeType.NEXT_SENT,
    "IN_BLOCK":  GraphEdgeType.IN_BLOCK,
    "IN_DOC":    GraphEdgeType.IN_DOC,
}
 
 
def build_claim_graph(
    claims: list[Claim],
    sir_doc: SIRDocument | None = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """
    Claim 리스트 → Knowledge Graph (노드 + 엣지) 조립.
 
    Args:
        claims:  Claim 객체 리스트 (schema 포함)
        sir_doc: SIRDocument (있으면 NEXT_SENT/IN_BLOCK/IN_DOC 문맥 엣지 추가)
 
    생성되는 노드/엣지:
      ClaimNode  → BELONGS_TO → MetricNode (indicator)
      ClaimNode  → MEASURED_AT → TimeNode  (time_period)
      ClaimNode  → BELONGS_TO → EntityNode (population)
      ClaimNode  ↔ COMPARE ↔  ClaimNode   (같은 indicator 공유 시)
      SentNode   → NEXT_SENT → SentNode   (문맥 순서)
      SentNode   → IN_BLOCK  → BlockNode  (문단 소속)
      BlockNode  → IN_DOC    → DocNode    (문서 소속)
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_node_ids: set[str] = set()
 
    def _add_node(node: GraphNode) -> None:
        if node.node_id not in seen_node_ids:
            nodes.append(node)
            seen_node_ids.add(node.node_id)
 
    # ── 1) 기본 Claim 노드/엣지 (기존 로직) ───────────────────────────
    # indicator별 claim_id 추적 (COMPARE 엣지 생성용)
    metric_to_claims: dict[str, list[str]] = defaultdict(list)
 
    for claim in claims:
        claim_node_id = f"claim:{claim.claim_id.hex[:8]}"
        _add_node(GraphNode(
            node_id=claim_node_id,
            node_type=GraphNodeType.CLAIM,
            label=claim.claim_text[:60],
            properties={
                "claim_text": claim.claim_text,
                "value": claim.schema.value if claim.schema else None,
                "claim_type": claim.claim_type,
                "canonical_type": (
                    claim.canonical_type.value
                    if isinstance(claim.canonical_type, ClaimType)
                    else claim.canonical_type
                ),
            },
        ))
 
        if not claim.schema:
            continue
 
        schema = claim.schema
 
        # MetricNode + BELONGS_TO
        if schema.indicator:
            metric_node_id = f"metric:{schema.indicator}"
            _add_node(GraphNode(
                node_id=metric_node_id,
                node_type=GraphNodeType.METRIC,
                label=schema.indicator,
                properties={"unit": schema.unit},
            ))
            edges.append(GraphEdge(
                from_node=claim_node_id,
                to_node=metric_node_id,
                edge_type=GraphEdgeType.BELONGS_TO,
            ))
            # COMPARE 엣지 대상 추적
            metric_to_claims[schema.indicator].append(claim_node_id)
 
        # TimeNode + MEASURED_AT
        if schema.time_period:
            time_node_id = f"time:{schema.time_period}"
            _add_node(GraphNode(
                node_id=time_node_id,
                node_type=GraphNodeType.TIME,
                label=schema.time_period,
            ))
            edges.append(GraphEdge(
                from_node=claim_node_id,
                to_node=time_node_id,
                edge_type=GraphEdgeType.MEASURED_AT,
            ))
 
        # EntityNode + BELONGS_TO
        if schema.population:
            entity_node_id = f"entity:{schema.population}"
            _add_node(GraphNode(
                node_id=entity_node_id,
                node_type=GraphNodeType.ENTITY,
                label=schema.population,
            ))
            edges.append(GraphEdge(
                from_node=claim_node_id,
                to_node=entity_node_id,
                edge_type=GraphEdgeType.BELONGS_TO,
            ))
 
    # ── 2) COMPARE 엣지 — 같은 indicator 공유 Claim 쌍 ────────────────
    #
    # 왜 필요한가:
    #   "쉬었음 청년이 20년 새 2.6배 늘었다" → 이건 C1(21만7천)+C2(8만4천) 비율
    #   두 Claim이 indicator="쉬었음인구"를 공유 → COMPARE 엣지로 연결
    #   검증 시 MetricNode 2-hop으로 C1+C2를 함께 KOSIS 조회 → 비율 계산
    #
    compare_added: set[tuple[str, str]] = set()
    for indicator, claim_ids in metric_to_claims.items():
        if len(claim_ids) < 2:
            continue
        for i in range(len(claim_ids)):
            for j in range(i + 1, len(claim_ids)):
                pair = (claim_ids[i], claim_ids[j])
                if pair not in compare_added:
                    edges.append(GraphEdge(
                        from_node=claim_ids[i],
                        to_node=claim_ids[j],
                        edge_type=GraphEdgeType.COMPARE,
                        properties={"shared_indicator": indicator},
                    ))
                    compare_added.add(pair)
 
    # ── 3) 문맥 엣지 — NEXT_SENT / IN_BLOCK / IN_DOC ──────────────────
    #
    # sir_doc이 있으면 extract_context_edges()로 문장-문단-문서 관계를 추가.
    # 이 엣지들이 없으면:
    #   - 같은 문단에 있는 수치들이 서로 연결되지 않음
    #   - "이 기사에서 취업 관련 수치 모두 검증" 쿼리 시
    #     Block B3 → IN_BLOCK → C7+C8+C9 탐색 불가
    #   - C1 다음 문장이 C2, 그 다음이 C3(2.6배) 관계를 모름
    #     → 파생 주장 자동 탐지 불가
    #
    if sir_doc is not None:
        from structverify.preprocessing.sir_builder import extract_context_edges
        context_edges = extract_context_edges(sir_doc)
 
        for ce in context_edges:
            edge_type = _CONTEXT_EDGE_MAP.get(ce["edge_type"])
            if edge_type is None:
                continue
            edges.append(GraphEdge(
                from_node=ce["from_node"],
                to_node=ce["to_node"],
                edge_type=edge_type,
            ))
 
        # 문맥 엣지에 등장하는 노드(블록/문장)도 등록
        # (Claim이 아닌 SentNode, BlockNode, DocNode)
        context_node_ids = set()
        for ce in context_edges:
            context_node_ids.add(ce["from_node"])
            context_node_ids.add(ce["to_node"])
        for nid in context_node_ids:
            if nid not in seen_node_ids:
                # node_type은 node_id prefix로 구분
                if nid.startswith("node:doc:"):
                    ntype = GraphNodeType.SOURCE
                elif nid.startswith("node:b"):
                    ntype = GraphNodeType.SOURCE  # Block → SOURCE로 임시 매핑
                else:
                    ntype = GraphNodeType.ENTITY
                _add_node(GraphNode(
                    node_id=nid,
                    node_type=ntype,
                    label=nid,
                ))
 
        logger.info(
            f"Graph 조립: {len(nodes)} nodes, {len(edges)} edges "
            f"(COMPARE: {len(compare_added)}쌍, 문맥엣지: {len(context_edges)}개)"
        )
    else:
        logger.info(f"Graph 조립: {len(nodes)} nodes, {len(edges)} edges (문맥엣지 없음 — sir_doc 미전달)")
 
    return nodes, edges