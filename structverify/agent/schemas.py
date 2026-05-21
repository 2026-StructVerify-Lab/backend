"""
structverify.agent.schemas — Agent 시스템의 데이터 모델.

핵심 개념:
  - DataPointSpec    : claim 검증에 필요한 *하나의 데이터 점* (지표 + 시점 + 인구)
  - Plan             : 검증 계획 — 어떤 데이터 점들이 필요한지 + 어떻게 찾을지
  - PlanStep         : 다음 행동 한 단위 (action + input + rationale)
  - Observation      : action 실행 결과
  - ReflectDecision  : Reflect Agent가 결정한 다음 행동
  - AgentVerdict     : Agent loop 최종 결과

Phase A에서는 *모델 정의*만. 실제 사용은:
  - Plan       → Phase C (Plan Agent)
  - Observation → Phase D (Loop)
  - ReflectDecision → Phase D (Reflect Agent)
  - AgentVerdict → Phase E (Verifier 확장)

사용 패턴:
    from structverify.agent.schemas import Plan, DataPointSpec, ClaimType

    plan = Plan(
        claim_id="abc-123",
        claim_type=ClaimType.COMPARISON,
        required_data=[
            DataPointSpec(indicator="평균소비성향", time="2014"),
            DataPointSpec(indicator="평균소비성향", time="2024"),
        ],
        initial_steps=[...],
    )
    workspace.write_plan(claim_id, plan.model_dump())
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────

class ClaimType(str, Enum):
    """Claim의 검증 유형 — Plan Agent가 분류한다."""

    ABSOLUTE = "absolute"
    """단일 시점 절대값. 예: '출생아 수 20,171명' (1개 데이터 점)."""

    DIFFERENCE = "difference"
    """두 시점 사이 차이값. 예: '0.04명 증가' (2개 데이터 점 + 차이 계산)."""

    GROWTH_RATE = "growth_rate"
    """두 시점 사이 증가율(%). 예: '6.7% 증가' (2개 데이터 점 + 비율 계산)."""

    COMPARISON = "comparison"
    """두 시점 직접 비교 (X→Y 형식). 예: '73.6% → 70.3%' (2개 데이터 점)."""

    RANKING = "ranking"
    """순위/상대 비교. 예: '1위', '하락폭이 가장 컸다' (여러 데이터 점)."""

    UNKNOWN = "unknown"
    """분류 불가."""


class ActionType(str, Enum):
    """Agent가 실행 가능한 action."""

    CATALOG_SEARCH = "catalog_search"
    """지정된 데이터 소스의 카탈로그(표 목록)에서 검색."""

    EXPLORE_CATALOG = "explore_catalog"
    """[패치 Q] 카탈로그의 카테고리 분포 + 대표 표를 탐색.

    LLM이 어떤 카테고리 어휘를 써야 할지 모를 때(예: '기상관측통계' vs '기후 변화')
    먼저 이 도구를 호출해 카탈로그가 실제로 어떤 분류·어휘를 쓰는지 파악한 뒤,
    그 정보를 바탕으로 정확한 catalog_search query/category를 만든다.
    룰베이스 도메인 매핑 없이도 DataSource 어휘를 자가학습."""

    FETCH_EVIDENCE = "fetch_evidence"
    """카탈로그에서 찾은 후보의 실제 데이터 조회."""

    READ_ORIGINAL = "read_original"
    """원문 기사 읽기 (claim 외 정보 필요할 때)."""

    CALCULATE = "calculate"
    """확보한 데이터로 수식 계산 (증가율, 차이 등)."""

    FINISH = "finish"
    """판정 확정 후 종료."""


class VerdictType(str, Enum):
    """Agent loop 결과 verdict."""
    MATCH = "match"
    MISMATCH = "mismatch"
    PARTIAL = "partial"
    UNVERIFIABLE = "unverifiable"


class StopReason(str, Enum):
    """Agent loop 종료 이유."""
    COMPLETED = "completed"
    """모든 데이터 확보 + 계산 + finish 결정."""

    MAX_ITERATIONS = "max_iterations"
    """max_iter 초과 — 미완료 상태로 종료."""

    GIVE_UP = "give_up"
    """agent가 명시적으로 포기 (충분히 시도했으나 데이터 없음)."""

    TOKEN_BUDGET = "token_budget"
    """LLM 토큰 예산 초과."""

    ERROR = "error"
    """예외 발생."""


# ── Data point ────────────────────────────────────────────────────

class DataPointSpec(BaseModel):
    """
    검증에 필요한 *하나의 데이터 점* 명세.

    Plan Agent가 claim에서 추출하고, agent loop이 *resolved_value*를 채움.
    """
    indicator: str
    """지표명. '출생아 수', '평균소비성향' 등."""

    time: str
    """시점. 'YYYY' 또는 'YYYY-MM' 형식 권장."""

    population: str | None = None
    """대상 인구. '전체', '30대 이하' 등."""

    unit_hint: str | None = None
    """단위 힌트. '%', '명', '원' (LLM이 추론 — 절대 강제 X)."""

    # ── 채워지는 필드 (agent loop 진행 중) ────────────────────────
    resolved_value: float | None = None
    """확보된 수치값. None이면 *아직 못 찾음*."""

    resolved_unit: str | None = None
    """실제 fetch된 데이터의 단위."""

    source: str | None = None
    """데이터 출처. 'KOSIS:DT_1L9U108', 'CSV:my_sales.csv' 등."""

    source_time: str | None = None
    """실제 fetch된 시점 (요청 시점과 다를 수 있음 — 가까운 row 매칭)."""


# ── Plan ──────────────────────────────────────────────────────────

class PlanStep(BaseModel):
    """Plan에 포함된 *예상 행동 단계*. Reflect Agent가 변경/건너뛸 수 있다."""

    action: ActionType
    input: dict[str, Any]
    rationale: str


class FallbackStrategy(BaseModel):
    """1차 탐색 실패 시 대안."""

    use_original_text: bool = False
    """원문에서 직접 수치 추출 시도."""

    alternative_keywords: list[str] = Field(default_factory=list)
    """다른 검색어 후보."""

    give_up_after_attempts: int = 5
    """이 횟수 시도 후에도 데이터 못 찾으면 unverifiable."""


class Plan(BaseModel):
    """Plan Agent가 생성하는 검증 계획."""

    claim_id: str
    claim_type: ClaimType
    required_data: list[DataPointSpec]
    initial_steps: list[PlanStep] = Field(default_factory=list)
    fallback: FallbackStrategy = Field(default_factory=FallbackStrategy)
    calculation_formula: str | None = None
    """예: '(current - prev) / prev * 100' (증가율), 'X - Y' (차이)."""

    notes: str | None = None
    """Plan Agent의 자유 형식 메모 (디버깅용)."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Observation + Reflect ────────────────────────────────────────

class Observation(BaseModel):
    """Action 실행 결과."""

    iter_num: int
    action: ActionType
    input: dict[str, Any]
    output: dict[str, Any]
    """Tool 호출 raw 결과. observations/*.json에도 저장됨."""

    summary: str
    """memory.md에 들어갈 한 줄 요약. agent가 다음 턴에 참고."""

    success: bool = True
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReflectDecision(BaseModel):
    """Reflect Agent의 다음 행동 결정."""

    thought: str
    """추론 과정 (디버깅 + 사용자에게 보일 수도)."""

    action: ActionType
    input: dict[str, Any]

    confidence_so_far: float = 0.0
    """0~1. finish 결정에 영향."""

    # finish action일 때만 채워짐
    proposed_verdict: VerdictType | None = None
    proposed_explanation: str | None = None


# ── Verdict ──────────────────────────────────────────────────────

class AgentVerdict(BaseModel):
    """Agent loop 최종 결과."""

    claim_id: str
    verdict: VerdictType
    confidence: float
    explanation: str

    data_points: list[DataPointSpec] = Field(default_factory=list)
    """확보된 데이터 점들. verifier.py가 이걸로 *최종 검산*."""

    iterations_used: int = 0
    total_tokens: int = 0
    stop_reason: StopReason = StopReason.COMPLETED

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
