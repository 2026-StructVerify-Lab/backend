"""
structverify.agent.tools.base — Tool 인터페이스 추상 클래스.

Tool = Agent가 호출 가능한 *원자 단위 작업*.
  - LLM이 reflect 단계에서 `{action: <ActionType>, input: {...}}` 결정
  - Loop이 해당 ActionType의 Tool 인스턴스 찾아서 execute()
  - 결과를 Observation으로 변환 → memory.md + log.jsonl + observations/*.json

설계 원칙:
  1. **추상화**: Tool은 *기존 코드 직접 import 금지*. ToolContext의 datasources/workspace만 사용.
     → KOSIS 외 source 추가 시 *Tool은 수정 X*, DataSource만 등록.
  2. **결과 표준화**: ToolResult 형식 통일. output dict + 요약 문자열 + 성공/실패.
  3. **Plugin 지원**: register_tool 데코레이터로 *동적 등록*. 회사 자체 Tool 추가 가능.

Phase B에서는 *5개 핵심 Tool* 구현:
  - catalog_search: DataSource 후보 검색
  - fetch_evidence: 데이터 조회
  - read_original: workspace의 원문 읽기
  - calculate: 안전한 수식 계산
  - finish: 종료 + Verdict 생성

Phase D (Loop)에서 이 Tool들을 *Reflect Agent가 선택해서 호출*한다.
"""
from __future__ import annotations

from structverify.utils.logger import get_logger
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type
from uuid import UUID

from pydantic import BaseModel, Field

from ..schemas import ActionType
from ..workspace import Workspace

logger = get_logger(__name__)


# ── 결과 모델 ────────────────────────────────────────────────────

class ToolResult(BaseModel):
    """Tool.execute()의 표준 반환 형식.

    Loop이 이걸 Observation으로 변환 (iter_num + action 추가).
    """

    output: dict[str, Any] = Field(default_factory=dict)
    """Tool 호출 raw 결과. observations/*.json에 저장됨."""

    summary: str = ""
    """memory.md에 들어갈 *한 줄 요약*. agent가 다음 턴에 참고."""

    success: bool = True
    error: str | None = None
    tokens_used: int = 0
    """이 호출에 쓴 LLM 토큰 (직접 LLM 호출하는 Tool만 채움)."""


# ── ToolContext — Tool에 주입되는 의존성 묶음 ──────────────────────

@dataclass
class ToolContext:
    """Tool 실행 시 필요한 *환경/의존성*. Loop이 만들어서 Tool에 전달.

    Tool은 self나 전역 상태 대신 *항상 context를 통해* 외부와 통신.
    이게 *추상화 + 테스트 용이성*의 핵심.
    """

    workspace: Workspace
    """현재 job의 workspace. read_source / append_memory / write_observation 등."""

    claim_id: str | UUID
    """현재 처리 중인 claim 식별자. workspace 파일 경로용."""

    config: dict[str, Any] = field(default_factory=dict)
    """Agent + DataSource + LLM 설정 dict. config.agent.*, config.data_sources.* 등."""

    datasources: dict[str, Any] = field(default_factory=dict)
    """{이름: BaseDataSource} 등록된 데이터 소스들. catalog_search/fetch tool이 사용."""

    iter_num: int = 0
    """현재 iteration 번호. 로깅 + observation 파일 이름용."""

    # 향후 추가 가능 (Phase D+):
    # llm_client: Any = None  # LLM 호출 클라이언트 (Reflect Agent용)
    # token_budget_remaining: int = 100_000


# ── ToolBase 추상 ────────────────────────────────────────────────

class ToolBase(ABC):
    """모든 Tool의 공통 인터페이스.

    필수 구현:
        - name: ActionType (CATALOG_SEARCH, FETCH_EVIDENCE, ...)
        - description: LLM에게 보여줄 설명 (Plan/Reflect Agent prompt에 삽입)
        - input_schema: 입력 형식 (LLM prompt에 표시)
        - execute(input_data, context): 실제 동작

    선택 구현:
        - validate_input(input_data): 호출 전 검증 (기본은 input_schema 키 존재만 확인)
    """

    name: ActionType
    description: str = ""
    """LLM이 이 Tool을 *언제 쓸지* 결정하도록 설명. Plan/Reflect prompt에 들어감."""

    input_schema: dict[str, Any] = {}
    """입력 dict의 *예상 키*와 *간단한 설명*.
    예: {"query": "검색어 (한국어 키워드)", "top_k": "최대 후보 수 (기본 5)"}
    """

    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Tool 실행. *예외는 ToolResult(success=False, error=...)로 변환*해서 반환.

        Loop은 ToolResult를 받아 Observation으로 wrap, memory에 기록.
        """

    def validate_input(self, input_data: dict[str, Any]) -> tuple[bool, str | None]:
        """입력 검증. (valid, error_msg). 기본 — input_schema의 키가 *모두* 있는지.

        하위 클래스에서 더 정교한 검증 override 가능.
        """
        missing = [k for k in self.input_schema if k not in input_data]
        if missing:
            return False, f"필수 입력 누락: {missing}"
        return True, None

    def render_help(self) -> str:
        """LLM prompt에 *이 Tool 설명*을 삽입할 때 사용. 사람-읽기 좋은 markdown."""
        lines = [f"### {self.name.value}", self.description, ""]
        if self.input_schema:
            lines.append("**Input fields:**")
            for k, desc in self.input_schema.items():
                lines.append(f"- `{k}`: {desc}")
        return "\n".join(lines)


# ── Tool Registry ────────────────────────────────────────────────

_TOOL_REGISTRY: dict[ActionType, Type[ToolBase]] = {}


def register_tool(action: ActionType) -> Callable[[Type[ToolBase]], Type[ToolBase]]:
    """Tool 클래스를 *ActionType*에 등록하는 데코레이터.

    Loop이 `tool_registry.get(action)`으로 Tool 클래스 찾고 인스턴스 만들어 execute.

    Usage:
        @register_tool(ActionType.CALCULATE)
        class CalculateTool(ToolBase):
            name = ActionType.CALCULATE
            description = "안전한 수식 계산"
            async def execute(...): ...
    """
    def decorator(cls: Type[ToolBase]) -> Type[ToolBase]:
        if not issubclass(cls, ToolBase):
            raise TypeError(f"{cls.__name__} must subclass ToolBase")
        if not getattr(cls, "name", None):
            cls.name = action
        if action in _TOOL_REGISTRY:
            logger.warning(
                f"[tools.registry] Tool '{action.value}' already registered, "
                f"overwriting ({_TOOL_REGISTRY[action].__name__} → {cls.__name__})"
            )
        _TOOL_REGISTRY[action] = cls
        logger.info(f"[tools.registry] Tool 등록: {action.value} ({cls.__name__})")
        return cls
    return decorator


def get_tool_class(action: ActionType) -> Type[ToolBase]:
    """등록된 Tool 클래스 반환. 없으면 KeyError."""
    if action not in _TOOL_REGISTRY:
        available = [a.value for a in _TOOL_REGISTRY]
        raise KeyError(
            f"Tool '{action.value}' not registered. Available: {available}"
        )
    return _TOOL_REGISTRY[action]


def list_tools() -> list[ActionType]:
    """등록된 모든 Tool action 목록."""
    return list(_TOOL_REGISTRY.keys())


def build_tool(action: ActionType) -> ToolBase:
    """Tool 인스턴스 생성 (기본은 인자 없음, override 가능)."""
    cls = get_tool_class(action)
    return cls()


def render_all_help(actions: list[ActionType] | None = None) -> str:
    """모든 등록된 Tool의 help를 markdown으로 — Plan/Reflect prompt에 삽입.

    Args:
        actions: 표시할 action 목록. None이면 *전부*.
    """
    if actions is None:
        actions = list(_TOOL_REGISTRY.keys())
    parts = []
    for a in actions:
        try:
            tool = build_tool(a)
            parts.append(tool.render_help())
        except Exception as e:
            logger.debug(f"render_help 실패 ({a.value}): {e}")
    return "\n\n".join(parts)
