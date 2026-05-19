# [이수민 - 2026-05-13]
# - AgentMemory에서 사용할 영속 데이터 모델 정의
# - MemoryNode/MemoryEdge: 기존 GraphNode/GraphEdge에 provenance 부착한 wrapper
# - Provenance: run_id + doc_id + 타임스탬프 (롤백·last-write-wins 용)
# - CrossDocEdgeType: cross-doc 전용 엣지(AGREES_WITH/SUPERSEDES/REUSED_BY)
#   ※ CONTRADICTS/SUPPORTS는 core/schemas.py:GraphEdgeType에 이미 존재 — 재사용
# - 하단에 canonical key 컨벤션(metric:{stat_id} 등) 주석으로 명시
"""
memory/schema.py — AgentMemory 영속 스키마

기존 GraphNode/GraphEdge를 감싸는 영속 wrapper.
- provenance 필드 추가 (어느 doc/run에서 생성됐는지)
- created_at / updated_at (last-write-wins용)
- cross-doc 전용 엣지 타입 enum
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from structverify.core.schemas import GraphNode, GraphEdge, GraphNodeType, GraphEdgeType


# ── Cross-doc 전용 엣지 타입 ─────────────────────────────────────────────────
# core/schemas.py의 GraphEdgeType은 단일 doc 그래프용.
# 여기는 누적 메모리에서 doc 간 관계를 표현하는 별도 enum.
#
# 주의: GraphEdgeType.CONTRADICTS / SUPPORTS는 이미 존재 — 재사용 가능.
# AgentMemory가 새로 만들 수 있는 doc 간 엣지는 아래 3종.

class CrossDocEdgeType(str, Enum):
    AGREES_WITH = "agrees_with"   # 같은 (Metric, Time, Entity)에 같은 value (다른 doc)
    SUPERSEDES  = "supersedes"    # 더 최근 doc이 같은 claim에 다른 값 (정정보도)
    REUSED_BY   = "reused_by"     # 과거 Evidence를 새 Claim 검증에 재활용


# ── Provenance 메타데이터 ───────────────────────────────────────────────────
# 모든 MemoryNode/MemoryEdge에 부착. 잘못된 doc 발견 시 그 기여분만 롤백 가능.

class Provenance(BaseModel):
    created_in_run: str    # run_id (파이프라인 1회 실행 단위)
    from_doc_id:    str    # doc_id (어느 문서에서 비롯됐는지)
    created_at:     datetime = Field(default_factory=datetime.utcnow)


# ── 영속 노드/엣지 ───────────────────────────────────────────────────────────

class MemoryNode(BaseModel):
    """
    JSONL에 저장될 영속 노드.

    GraphNode를 감싸고 provenance + 타임스탬프 추가.
    공유 노드(Metric/Entity/Time/Source/Evidence)는 canonical key로 dedup,
    문서별 노드(Document/Claim/Sentence)는 append-only.
    """
    # 기존 GraphNode 필드
    node_id:    str                # canonical key (e.g. "metric:DT_1ES4001")
    node_type:  GraphNodeType
    label:      str
    domain:     str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    # 영속화용 필드
    provenance: Provenance
    updated_at: datetime = Field(default_factory=datetime.utcnow)  # last-write-wins

    @classmethod
    def from_graph_node(
        cls,
        node: GraphNode,
        run_id: str,
        doc_id: str,
    ) -> "MemoryNode":
        """런타임 GraphNode → 영속 MemoryNode 변환."""
        raise NotImplementedError  # Phase 2 구현


class MemoryEdge(BaseModel):
    """
    JSONL에 저장될 영속 엣지.

    GraphEdge를 감싸고 provenance 추가.
    edge_type은 GraphEdgeType 또는 CrossDocEdgeType.
    """
    edge_id:   str
    from_node: str
    to_node:   str
    # edge_type은 GraphEdgeType.value 또는 CrossDocEdgeType.value 문자열
    edge_type: str
    weight:    float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)

    # 영속화용 필드
    provenance: Provenance

    @classmethod
    def from_graph_edge(
        cls,
        edge: GraphEdge,
        run_id: str,
        doc_id: str,
    ) -> "MemoryEdge":
        """런타임 GraphEdge → 영속 MemoryEdge 변환."""
        raise NotImplementedError  # Phase 2 구현


# ── Canonical key 컨벤션 ────────────────────────────────────────────────────
# 공유 노드의 node_id 형식. Phase 2에서 실제 생성 시 이 규칙 따름.
#
#   Metric    : "metric:{stat_id}"          예) "metric:DT_1ES4001"
#   Source    : "source:{stat_id}"          예) "source:DT_1ES4001"
#   Evidence  : "evidence:{stat_id}:{time}" 예) "evidence:DT_1ES4001:2024"
#   Entity    : "entity:{normalized_name}"  예) "entity:청년"
#   Time      : "time:{iso}"                예) "time:2024", "time:2024-09"
#
# 문서별 노드(Document/Claim/Sentence)는 기존 UUID/anchor_id 사용.
