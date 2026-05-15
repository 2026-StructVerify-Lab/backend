# [이수민 - 2026-05-13]
# - JSONL append-only 저장소 (Phase 1)
# - 저장 위치:
#     backend/structverify/memory/nodes/{type}.jsonl  (노드 타입별 분리)
#     backend/structverify/memory/edges/edges.jsonl   (엣지는 단일 파일)
# - 원칙: append-only, 수정/삭제 안 함. dedup은 in-memory load 시 last-write-wins
# - 노출 함수: read_all_nodes / read_all_edges / append_node / append_edge
# - _NODE_FILES dict로 GraphNodeType → 파일명 매핑
# - 함수 본문은 Phase 2 구현 (현재는 시그니처+docstring만)
"""
memory/storage/jsonl_store.py — JSONL append-only 저장소 (Phase 1)

[저장 위치]
  backend/structverify/memory/nodes/{type}.jsonl
  backend/structverify/memory/edges/edges.jsonl

[원칙]
- Append-only: 수정/삭제 안 함. dedup은 in-memory load 시 last-write-wins.
- 노드 타입별 분리: metrics / entities / times / sources / evidences /
  documents / claims / temporals
- 엣지는 단일 파일

[Phase 1 — 구조화]
함수 시그니처만 정의. 본문은 Phase 2에서 구현.
"""
from __future__ import annotations

from pathlib import Path

from structverify.memory.schema import MemoryEdge, MemoryNode
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 노드 파일 이름 컨벤션 ───────────────────────────────────────────────────
# GraphNodeType → JSONL 파일명 매핑
_NODE_FILES = {
    "metric":        "metrics.jsonl",
    "entity":        "entities.jsonl",
    "time":          "times.jsonl",
    "source":        "sources.jsonl",
    "evidence":      "evidences.jsonl",
    "document":      "documents.jsonl",
    "claim":         "claims.jsonl",
    "sentence":      "sentences.jsonl",
    "temporal_expr": "temporals.jsonl",
    "resolved_time": "temporals.jsonl",
}
_EDGES_FILE = "edges.jsonl"


# ── 읽기 ────────────────────────────────────────────────────────────────────

def read_all_nodes(nodes_dir: Path) -> list[MemoryNode]:
    """
    nodes/ 디렉터리의 모든 .jsonl 파일을 읽어 MemoryNode 리스트로 반환.

    같은 node_id가 여러 번 등장하면 마지막 것 채택 (last-write-wins).

    Args:
        nodes_dir: Path to backend/structverify/memory/nodes/

    Returns:
        list[MemoryNode]
    """
    raise NotImplementedError  # Phase 2


def read_all_edges(edges_dir: Path) -> list[MemoryEdge]:
    """
    edges/edges.jsonl을 읽어 MemoryEdge 리스트로 반환.

    edge는 dedup 안 함 — 같은 (from, to, type)이라도 다른 doc/run에서 왔으면
    별개 엣지로 보존. (그래야 provenance 추적 가능)

    Args:
        edges_dir: Path to backend/structverify/memory/edges/

    Returns:
        list[MemoryEdge]
    """
    raise NotImplementedError  # Phase 2


# ── 쓰기 (append) ───────────────────────────────────────────────────────────

def append_node(node: MemoryNode, nodes_dir: Path) -> None:
    """
    MemoryNode를 적절한 타입별 JSONL 파일에 append.

    파일 결정:
        node.node_type.value → _NODE_FILES[...]

    Args:
        node:      append할 MemoryNode
        nodes_dir: Path to backend/structverify/memory/nodes/
    """
    raise NotImplementedError  # Phase 2


def append_edge(edge: MemoryEdge, edges_dir: Path) -> None:
    """
    MemoryEdge를 edges/edges.jsonl에 append.

    Args:
        edge:      append할 MemoryEdge
        edges_dir: Path to backend/structverify/memory/edges/
    """
    raise NotImplementedError  # Phase 2


# ── 유틸 ────────────────────────────────────────────────────────────────────

def ensure_dirs(memory_dir: Path) -> None:
    """nodes/, edges/ 디렉터리가 없으면 생성."""
    raise NotImplementedError  # Phase 2
