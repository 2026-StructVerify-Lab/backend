"""
graph/claim_graph.py — ClaimGraph facade (멀티홉 traversal API)

[설계 의도]
graph_builder가 만든 노드/엣지 list를 다운스트림(schema_inductor, verifier,
explainer)이 쉽게 traverse할 수 있도록 감싸는 lookup 객체.

다운스트림은 그래프 내부 구조(어떤 엣지가 어디서 나가는지)를 몰라도
high-level API만 호출하면 멀티홉 결과를 얻는다.

예:
    graph = ClaimGraph(nodes, edges)
    resolved = graph.resolve_time_for_claim(claim)
    # 내부적으로:
    # Claim → HAS_TEMPORAL → TemporalExpr → RESOLVES_TO → ResolvedTime
    # 또는
    # Claim → IN_BLOCK → Sentence → HAS_TEMPORAL → TemporalExpr → RESOLVES_TO

이 facade가 있어야 verifier/schema_inductor가 그래프를 1급 입력으로 받게 된다.
지금 코드의 가장 큰 문제(그래프를 만들어놓고 안 읽음)를 해결하는 핵심 인터페이스.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from structverify.core.schemas import (
    Claim, GraphEdge, GraphEdgeType, GraphNode, GraphNodeType,
)


class ClaimGraph:
    """
    Claim/Document/Temporal 그래프 위에서 멀티홉 쿼리를 수행하는 facade.

    인스턴스 하나가 한 문서에 대응. 노드/엣지는 frozen 상태로 가정.
    """

    def __init__(self, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]):
        self._nodes: dict[str, GraphNode] = {n.node_id: n for n in nodes}

        # 인접 리스트: from_node → [(edge_type, to_node), ...]
        self._out: dict[str, list[tuple[GraphEdgeType, str]]] = defaultdict(list)
        # 역방향: to_node → [(edge_type, from_node), ...]
        self._in: dict[str, list[tuple[GraphEdgeType, str]]] = defaultdict(list)

        for e in edges:
            self._out[e.from_node].append((e.edge_type, e.to_node))
            self._in[e.to_node].append((e.edge_type, e.from_node))

    # ── 기본 lookup ─────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def neighbors(
        self,
        node_id: str,
        edge_type: GraphEdgeType | None = None,
    ) -> list[str]:
        """node_id의 outgoing 이웃. edge_type 필터 옵션."""
        out = self._out.get(node_id, [])
        if edge_type is None:
            return [t for _, t in out]
        return [t for et, t in out if et == edge_type]

    def in_neighbors(
        self,
        node_id: str,
        edge_type: GraphEdgeType | None = None,
    ) -> list[str]:
        """node_id의 incoming 이웃."""
        ins = self._in.get(node_id, [])
        if edge_type is None:
            return [f for _, f in ins]
        return [f for et, f in ins if et == edge_type]

    # ── 문서 레벨 ───────────────────────────────────────────────────────────

    def find_document(self) -> GraphNode | None:
        """문서 노드 찾기. 문서당 하나만 있다고 가정."""
        for n in self._nodes.values():
            if n.node_type == GraphNodeType.DOCUMENT:
                return n
        return None

    def get_anchor_year(self) -> int | None:
        """문서의 anchor_year 속성. 없으면 None."""
        doc = self.find_document()
        if not doc:
            return None
        return doc.properties.get("anchor_year")

    # ── 시간 해소 (멀티홉 핵심) ─────────────────────────────────────────────

    def resolve_time_for_sentence(self, sent_anchor_id: str) -> str | None:
        """
        문장 노드에 연결된 TemporalExpr 중 가장 먼저 발견되는 resolved 값 반환.

        Traversal:
          SentenceNode ─HAS_TEMPORAL→ TemporalExpr ─RESOLVES_TO→ ResolvedTime

        여러 표현이 있으면 첫 번째 것 사용 (LLM이 추출한 순서).
        """
        for te_id in self.neighbors(sent_anchor_id, GraphEdgeType.HAS_TEMPORAL):
            resolved = self._resolved_value_of(te_id)
            if resolved:
                return resolved
        return None

    def resolve_time_for_claim(self, claim: Claim) -> str | None:
        """
        Claim의 시점 해소.

        Step 1: claim의 sent_id에 연결된 TemporalExpr 우선 사용
        Step 2: 그게 REFERS_TO로 다른 문장을 가리키면 그 문장으로 다시 traverse
        """
        sent_anchor = self._claim_to_sent_anchor(claim)
        if not sent_anchor:
            return None

        # 1차: 직접 연결된 TemporalExpr
        for te_id in self.neighbors(sent_anchor, GraphEdgeType.HAS_TEMPORAL):
            # REFERS_TO가 있으면 참조 대상 문장의 시점도 시도
            ref_sent = self._first_neighbor(te_id, GraphEdgeType.REFERS_TO)

            # 우선 자체 resolved 시도
            direct = self._resolved_value_of(te_id)
            if direct:
                return direct

            # 자체로 안 풀리면 참조 문장으로 hop
            if ref_sent:
                ref_resolved = self.resolve_time_for_sentence(ref_sent)
                if ref_resolved:
                    return ref_resolved

        return None

    def count_temporal_expressions(self, claim: Claim) -> int:
        """[v6.19] claim의 문장에 달린 TemporalExpr 개수.

        한 문장에 "작년"+"재작년"처럼 시간표현이 여러 개면,
        temporal_provenance가 어느 것이 이 claim에 맞는지 구분할 수
        없다(첫 번째를 무조건 반환). 이 경우 schema_inductor가
        단정적 hint 대신 anchor 변환표만 받아 LLM이 claim 문맥으로
        직접 고르게 해야 한다.
        """
        sent_anchor = self._claim_to_sent_anchor(claim)
        if not sent_anchor:
            return 0
        return len(self.neighbors(sent_anchor, GraphEdgeType.HAS_TEMPORAL))

    def temporal_provenance(self, claim: Claim) -> dict | None:
        """
        시점 해소의 근거 (디버깅/설명용).

        Returns:
            {
              "expression": "재작년 같은 기간",
              "resolved": "2022-01..2022-11",
              "basis": "anchor_year - 2 + s0009의 기간 참조",
              "via_coref": "node:s0009",
            }
        """
        sent_anchor = self._claim_to_sent_anchor(claim)
        if not sent_anchor:
            return None

        for te_id in self.neighbors(sent_anchor, GraphEdgeType.HAS_TEMPORAL):
            te_node = self._nodes.get(te_id)
            if not te_node:
                continue

            resolved = self._resolved_value_of(te_id)
            ref_sent = self._first_neighbor(te_id, GraphEdgeType.REFERS_TO)

            if not resolved and ref_sent:
                resolved = self.resolve_time_for_sentence(ref_sent)

            if resolved:
                return {
                    "expression": te_node.properties.get("expression"),
                    "resolved": resolved,
                    "basis": te_node.properties.get("resolution_basis"),
                    "via_coref": ref_sent,
                }

        return None

    # ── 내부 traversal ──────────────────────────────────────────────────────

    def _resolved_value_of(self, temporal_expr_node_id: str) -> str | None:
        """TemporalExpr 노드에서 RESOLVES_TO로 연결된 ResolvedTime의 value."""
        for rt_id in self.neighbors(temporal_expr_node_id, GraphEdgeType.RESOLVES_TO):
            rt_node = self._nodes.get(rt_id)
            if rt_node:
                return rt_node.properties.get("value") or rt_node.label
        return None

    def _first_neighbor(
        self, node_id: str, edge_type: GraphEdgeType,
    ) -> str | None:
        nbrs = self.neighbors(node_id, edge_type)
        return nbrs[0] if nbrs else None

    def _claim_to_sent_anchor(self, claim: Claim) -> str | None:
        """Claim → 문장 anchor id."""
        return claim.graph_anchor_id

    # ── 디버깅 ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "nodes": len(self._nodes),
            "edges": sum(len(v) for v in self._out.values()),
            "node_types": self._count_by("node_type"),
        }

    def _count_by(self, attr: str) -> dict:
        out: dict = defaultdict(int)
        for n in self._nodes.values():
            v = getattr(n, attr, None)
            key = v.value if hasattr(v, "value") else str(v)
            out[key] += 1
        return dict(out)