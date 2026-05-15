"""
agent/context.py — Claim 단위 RunContext (에이전틱 루프 메모리)

한 claim의 Step 5~9 실행 동안 살아있는 in-memory 컨텍스트.
Planner/Critic이 판단할 때 이 객체를 참조한다.

- 담당자: 신준수
"""
# 수정자: 신준수
# 수정 날짜: 2026-05-15
# 수정 내용: 에이전틱 리팩토링 - RunContext / StepSnapshot / CriticVerdict 신규 정의

# [DONE] CriticVerdict enum 정의
# [DONE] StepSnapshot dataclass 정의
# [DONE] RunContext dataclass 정의
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from structverify.core.schemas import Claim, GraphEdge, GraphNode


# ── CriticVerdict ────────────────────────────────────────────────────────────

class CriticVerdict(str, Enum):
    """
    Critic.evaluate()가 반환하는 판정값.

    OK         : 결과 정상 → 다음 스텝으로 진행
    RETRY_SAME : 같은 스텝을 파라미터 그대로 재시도 (일시적 실패)
    ROLLBACK   : 상위 스텝으로 돌아가야 함 (Planner가 방향 결정)
    STOP       : 롤백 없이 종료 (예: MISMATCH — 진짜 틀린 것)
    GIVE_UP    : 이 claim 검증 포기 → UNVERIFIABLE 확정
    """
    OK         = "ok"
    RETRY_SAME = "retry_same"
    ROLLBACK   = "rollback"
    STOP       = "stop"
    GIVE_UP    = "give_up"


# ── StepSnapshot ────────────────────────────────────────────────────────────

@dataclass
class StepSnapshot:
    """
    스텝 실행 결과 스냅샷.
    롤백 시 이전 상태 참조 및 Planner 판단 근거로 사용.
    """
    step:          int
    output:        Any
    strategy:      dict        # Planner가 결정한 전략 힌트
    success:       bool
    failed_reason: str | None = None  # Critic이 판단한 실패 원인


# ── RunContext ───────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    """
    Claim 단위 에이전틱 루프 컨텍스트.

    DocumentWorkingMemory(문서 단위)와 다른 layer:
      - DocumentWorkingMemory: 문서 전체 공유, 정확도용 (도메인 가드, stat 캐시)
      - RunContext: claim 전용, 에이전틱 제어용 (롤백, 재시도, 힌트)

    local_nodes / local_edges:
      공유 리스트에 즉시 append하지 않고 여기에 쌓아두다가
      루프 성공 시에만 all_nodes/all_edges에 merge한다. (T1 해결)
    """
    claim:         Claim

    # ── 시도 카운터 ──────────────────────────────────────────────────────────
    attempt_count: int  = 0
    max_attempts:  int  = 3  # 롤백 최대 횟수 (초과 시 GIVE_UP)

    # ── 스텝별 결과 스냅샷 ──────────────────────────────────────────────────
    snapshots:     dict[int, StepSnapshot] = field(default_factory=dict)

    # ── 롤백 이력 ────────────────────────────────────────────────────────────
    # Planner.plan_rollback()이 반환한 dict가 여기에 쌓임
    # 예: [{"rollback_to": 5, "reason": "indicator 너무 구체적", "hint": "...", "give_up": False}]
    rollback_log:  list[dict] = field(default_factory=list)

    # ── 스텝별 재시도 힌트 ──────────────────────────────────────────────────
    # Planner가 결정한 힌트. Executor가 해당 스텝 실행 시 레이어 함수에 전달.
    # 예: hints[5] = "indicator='쉬었음 인구', source_phrase='21만7천명' 타겟팅"
    hints:         dict[int, str] = field(default_factory=dict)

    # ── 로컬 노드/엣지 버퍼 (T1 해결) ──────────────────────────────────────
    # 성공 시에만 all_nodes/all_edges에 merge
    local_nodes:   list[GraphNode] = field(default_factory=list)
    local_edges:   list[GraphEdge] = field(default_factory=list)

    # ── 편의 메서드 ──────────────────────────────────────────────────────────

    def record_snapshot(
        self,
        step: int,
        output: Any,
        strategy: dict | None = None,
        success: bool = True,
        failed_reason: str | None = None,
    ) -> None:
        """스텝 실행 결과를 snapshots에 기록."""
        self.snapshots[step] = StepSnapshot(
            step=step,
            output=output,
            strategy=strategy or {},
            success=success,
            failed_reason=failed_reason,
        )

    def record_rollback(self, plan: dict) -> None:
        """
        Planner.plan_rollback() 결과를 기록하고 attempt_count 증가.
        plan: {"rollback_to": int, "reason": str, "hint": str, "give_up": bool}
        """
        self.rollback_log.append(plan)
        self.attempt_count += 1
        rollback_to = plan.get("rollback_to")
        hint = plan.get("hint", "")
        if rollback_to is not None and hint:
            self.hints[rollback_to] = hint

    def is_exhausted(self) -> bool:
        """attempt_count가 max_attempts에 도달했으면 True."""
        return self.attempt_count >= self.max_attempts

    def last_snapshot(self, step: int) -> StepSnapshot | None:
        """특정 스텝의 마지막 스냅샷 반환."""
        return self.snapshots.get(step)

    def retry_count_for_step(self, step: int) -> int:
        """
        특정 스텝에서 RETRY_SAME이 몇 번 발생했는지 롤백 로그에서 추산.
        (롤백 로그에 같은 step에 대한 retry 기록이 있는지 확인)
        """
        return sum(
            1 for log in self.rollback_log
            if log.get("rollback_to") == step
        )

    def clear_local_buffers(self) -> None:
        """롤백 시 로컬 노드/엣지 버퍼 초기화."""
        self.local_nodes.clear()
        self.local_edges.clear()
