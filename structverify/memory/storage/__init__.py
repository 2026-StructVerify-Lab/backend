# [이수민 - 2026-05-13]
# - memory.storage 패키지 진입점 (I/O 전담 layer)
# - 로직(agent_memory.py) / 저장(storage/)을 분리하는 경계
# - 현재 노출: read_all_nodes / read_all_edges / append_node / append_edge
# - Phase 2 마이그레이션 시 jsonl_store → neo4j_store로 교체
#   agent_memory.py 인터페이스는 그대로 유지
"""
memory.storage — 영속 layer (Phase 1: JSONL, Phase 2: Neo4j)

Phase 1 책임 분리:
  agent_memory.py → 로직 (dedup, merge, query)
  storage/        → I/O (read_all, append)

Phase 2 마이그레이션 시 jsonl_store를 neo4j_store로 교체.
agent_memory.py의 인터페이스는 그대로.
"""
from structverify.memory.storage.jsonl_store import (
    read_all_nodes,
    read_all_edges,
    append_node,
    append_edge,
)

__all__ = [
    "read_all_nodes",
    "read_all_edges",
    "append_node",
    "append_edge",
]
