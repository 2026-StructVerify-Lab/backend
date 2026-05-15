# [이수민 - 2026-05-13]
# - 누적 KG 영속화 + dedup의 단일 진입점 (facade)
# - load() : 파이프라인 시작 시 JSONL 전체 → ClaimGraph 로 복원
# - merge(): Step 7 종료 후 호출, 신규 노드/엣지를 누적 메모리에 머지
#            · 공유 노드(Metric/Entity/Time/Source/Evidence) dedup
#            · 문서별 노드(Document/Claim/Sentence) append
#            · 같은 (Metric, Time, Entity) 다른 value → CONTRADICTS edge 자동 생성
# - query_metric() / query_evidence(): 과거 검증 결과 재활용용 조회 API
# - 메서드 본문은 Phase 2에서 구현 (현재는 시그니처+docstring만)
"""
memory/agent_memory.py — AgentMemory facade

누적 KG의 load / merge / query를 담당.
파이프라인 시작 시 load(), Step 7 종료 후 merge() 호출.

[Phase 1 — 구조화]
메서드 시그니처만 정의. 본문은 Phase 2에서 구현.

[Phase 2 — 구현 예정]
- load(): JSONL 전부 읽어 ClaimGraph 인스턴스 구성
- merge(): 신규 노드/엣지 머지 (dedup + CONTRADICTS 자동 탐지)
- query_metric(): 특정 Metric에 매달린 과거 Claim 조회

[Phase 3 — 통합 예정]
runtime_agent.py:
  Step 0  memory = AgentMemory(); graph = await memory.load()
  Step 7 이후  await memory.merge(all_nodes, all_edges, doc_id, run_id)
"""
from __future__ import annotations

from pathlib import Path

from structverify.core.schemas import Claim, GraphEdge, GraphNode
from structverify.graph.claim_graph import ClaimGraph
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class AgentMemory:
    """
    누적 KG의 영속화 + dedup layer.

    [라이프사이클]
        memory = AgentMemory()
        graph  = await memory.load()           # Step 0
        ...
        await memory.merge(nodes, edges, ...)  # Step 7 이후
    """

    def __init__(
        self,
        memory_dir: str | Path = "backend/structverify/memory",
    ):
        """
        Args:
            memory_dir: JSONL 데이터 루트. 하위에 nodes/ edges/ 디렉터리 존재.
        """
        self.memory_dir = Path(memory_dir)
        self.nodes_dir = self.memory_dir / "nodes"
        self.edges_dir = self.memory_dir / "edges"

    # ── 1. 로드 ──────────────────────────────────────────────────────────────

    async def load(self) -> ClaimGraph:
        """
        JSONL 전체 → ClaimGraph 인스턴스.

        파이프라인 시작 시 호출. 누적된 과거 KG를 인-메모리로 올림.

        Returns:
            ClaimGraph: 모든 과거 노드/엣지가 포함된 facade
        """
        raise NotImplementedError  # Phase 2

    # ── 2. 머지 (Step 7 이후) ────────────────────────────────────────────────

    async def merge(
        self,
        new_nodes: list[GraphNode],
        new_edges: list[GraphEdge],
        doc_id: str,
        run_id: str,
    ) -> dict:
        """
        신규 그래프를 누적 메모리에 머지.

        흐름:
          ① 공유 노드(Metric/Entity/Time/Source/Evidence) dedup by canonical_id
          ② 문서별 노드(Document/Claim/Sentence) append
          ③ 같은 (Metric, Time, Entity)에 다른 value 발견 → CONTRADICTS edge
          ④ MemoryNode/MemoryEdge로 wrap (provenance 부착)
          ⑤ JSONL append

        Args:
            new_nodes: 이번 doc의 신규 노드
            new_edges: 이번 doc의 신규 엣지
            doc_id:    이번 doc의 식별자
            run_id:    이번 파이프라인 실행 식별자

        Returns:
            dict: {
                "added_nodes":     int,
                "deduped_nodes":   int,
                "added_edges":     int,
                "contradicts_found": int,
            }
        """
        raise NotImplementedError  # Phase 2

    # ── 3. 쿼리 ──────────────────────────────────────────────────────────────

    async def query_metric(self, canonical_id: str) -> list[Claim]:
        """
        특정 Metric에 매달린 모든 과거 Claim 조회.

        verifier가 새 claim 검증 전, 같은 indicator의 과거 verified claim이
        있는지 확인할 때 사용. cache hit이면 KOSIS 재조회 생략 가능.

        Args:
            canonical_id: "metric:DT_1ES4001" 형식

        Returns:
            list[Claim]: 그 metric에 매달린 모든 Claim (verified 우선)
        """
        raise NotImplementedError  # Phase 2

    async def query_evidence(self, stat_id: str, time_period: str | None = None):
        """
        특정 stat_id + 시점의 Evidence 캐시 조회.

        같은 KOSIS row를 다른 doc이 이미 조회했다면 그 결과 재활용.
        REUSED_BY 엣지 생성용.

        Args:
            stat_id:     KOSIS 통계표 ID (e.g. "DT_1ES4001")
            time_period: 시점 (None이면 stat_id의 모든 시점)

        Returns:
            Evidence | None
        """
        raise NotImplementedError  # Phase 2
