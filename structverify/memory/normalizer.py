# [이수민 - 2026-05-13]
# - C안(KOSIS stat_id 기반) 정규화 헬퍼 모음
# - canonicalize_metric/source/evidence: stat_id → canonical node_id 문자열 생성
#   · metric:{stat_id}        예) "metric:DT_1ES4001"
#   · source:{stat_id}
#   · evidence:{stat_id}:{time}   예) "evidence:DT_1ES4001:2024"
# - rename_metric_to_canonical(): Step 7 이후 호출
#   · 임시 Metric 노드(LLM이 뱉은 indicator명) → canonical id로 rename
#   · 매달려있던 BELONGS_TO 등 엣지를 canonical 노드로 redirect
# - 함수 본문은 Phase 2 구현 (현재는 시그니처+docstring만)
"""
memory/normalizer.py — Canonical key 생성

C안 (KOSIS stat_id 기반 정규화):
  임시 Metric 노드 (LLM이 뱉은 indicator명)
    → Evidence 도착 후 stat_id로 canonical node_id 생성
    → 누적 메모리의 동일 stat_id Metric과 자동 dedup

[Phase 1 — 구조화]
함수 시그니처만 정의. 본문은 Phase 2에서 구현.
"""
from __future__ import annotations

from structverify.core.schemas import Evidence, GraphEdge, GraphNode
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── Canonical id 생성 ────────────────────────────────────────────────────────

def canonicalize_metric(stat_id: str) -> str:
    """
    KOSIS stat_id → Metric 노드의 canonical node_id.

    예시:
        canonicalize_metric("DT_1ES4001") → "metric:DT_1ES4001"

    Args:
        stat_id: KOSIS 통계표 ID (DT_로 시작하는 문자열)

    Returns:
        canonical node_id 문자열
    """
    raise NotImplementedError  # Phase 2


def canonicalize_source(stat_id: str) -> str:
    """KOSIS stat_id → Source 노드의 canonical node_id."""
    raise NotImplementedError  # Phase 2


def canonicalize_evidence(stat_id: str, time_period: str | None) -> str:
    """
    Evidence 노드의 canonical node_id.

    같은 stat_id라도 시점이 다르면 다른 Evidence.

    예시:
        canonicalize_evidence("DT_1ES4001", "2024")    → "evidence:DT_1ES4001:2024"
        canonicalize_evidence("DT_1ES4001", "2024-09") → "evidence:DT_1ES4001:2024-09"
    """
    raise NotImplementedError  # Phase 2


# ── Metric 노드 rename + 엣지 redirect ───────────────────────────────────────

def rename_metric_to_canonical(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    evidence: Evidence,
    temp_metric_id: str,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """
    Step 7 이후 호출. 임시 Metric 노드를 canonical id로 rename + edge redirect.

    흐름:
      ① evidence.stat_table_id로 canonical id 계산
      ② nodes에서 temp_metric_id를 찾아 node_id를 canonical로 교체
      ③ edges에서 from_node/to_node가 temp_metric_id인 것을 canonical로 redirect

    Args:
        nodes:          Step 6에서 만든 노드 리스트 (in-place 수정 또는 copy)
        edges:          Step 6에서 만든 엣지 리스트
        evidence:       Step 7에서 받은 Evidence (stat_table_id 보유)
        temp_metric_id: LLM이 뱉은 indicator 기반 임시 id (e.g. "metric:쉬었음인구")

    Returns:
        (정규화된 nodes, 정규화된 edges)
    """
    raise NotImplementedError  # Phase 2
