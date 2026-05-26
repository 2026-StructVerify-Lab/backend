"""
structverify.agent.tools — Agent가 호출하는 Tool 모음.

각 Tool은 *@register_tool(ActionType.X)* 데코레이터로 자동 등록됨.
이 패키지를 import하면 *모든 Tool이 registry에 등록됨*.

Phase B Tool 목록:
  - catalog_search  : DataSource 카탈로그(표) 검색
  - fetch_evidence  : 후보의 실제 수치 조회
  - read_original   : workspace의 원문 기사 읽기
  - calculate       : 안전한 수식 계산
  - finish          : 검증 종료 + Verdict 생성

회사 자체 Tool 추가:
    from structverify.agent.tools import register_tool, ToolBase
    from structverify.agent.schemas import ActionType

    # ActionType에 새 항목 추가 후 (또는 기존 사용):
    @register_tool(ActionType.YOUR_ACTION)
    class YourTool(ToolBase):
        ...

Usage (Phase D Loop에서):
    from structverify.agent.tools import get_tool_class, list_tools, render_all_help

    # LLM prompt에 모든 Tool 설명 삽입
    prompt = f"... Available tools:\n{render_all_help()} ..."

    # LLM 응답 (decision)에서 action 받아서 실행
    tool_cls = get_tool_class(decision.action)
    tool = tool_cls()
    result = await tool.execute(decision.input, context)
"""
from .base import (
    ToolBase,
    ToolContext,
    ToolResult,
    register_tool,
    get_tool_class,
    list_tools,
    build_tool,
    render_all_help,
)

# 모든 Tool 모듈 import — register_tool 데코레이터 실행 트리거.
# (import 순서 무관 — registry는 ActionType 키 기반)
from . import calculate           # noqa: F401
from . import finish              # noqa: F401
from . import read_original       # noqa: F401
from . import catalog_search      # noqa: F401
from . import fetch_evidence      # noqa: F401
from . import explore_catalog     # noqa: F401
from . import replan              # noqa: F401

__all__ = [
    "ToolBase",
    "ToolContext",
    "ToolResult",
    "register_tool",
    "get_tool_class",
    "list_tools",
    "build_tool",
    "render_all_help",
]
