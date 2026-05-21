"""
structverify.agent.tools.calculate — 안전한 수식 계산 Tool.

Agent가 *데이터 점 N개로 수식 계산*할 때 사용.

예시 사용:
  - 증가율: (current - prev) / prev * 100
  - 차이: current - prev
  - 비교: a / b * 100 (점유율)

안전성:
  - `eval()`은 *절대 사용 X*.
  - AST 기반 *제한된 산술 평가*. 식별자(변수) + 숫자 + 4칙 연산 + 비교만.
  - 함수 호출/속성 접근/import 등 일체 차단.

변수 주입:
  - input_data["variables"] = {"current": 20717, "prev": 19059}
  - expression = "(current - prev) / prev * 100"
  - → 결과: 8.7028...
"""
from __future__ import annotations

import ast
from structverify.utils.logger import get_logger
import math
import operator
from typing import Any

from ..schemas import ActionType
from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


# ── 허용된 AST 노드 + 연산 ─────────────────────────────────────────

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# 안전한 math 함수 (필요 시 추가)
_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
}


def _safe_eval(node: ast.AST, variables: dict[str, float]) -> float | int | bool:
    """제한된 AST 평가."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables)

    # 숫자 상수
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"허용되지 않는 상수: {type(node.value).__name__}")

    # 변수 (식별자)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.id]
        raise NameError(f"정의되지 않은 변수: {node.id!r}")

    # 이항 연산
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"허용되지 않는 연산: {op_type.__name__}")
        left = _safe_eval(node.left, variables)
        right = _safe_eval(node.right, variables)
        return _ALLOWED_BIN_OPS[op_type](left, right)

    # 단항 연산 (-x, +x)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"허용되지 않는 단항 연산: {op_type.__name__}")
        return _ALLOWED_UNARY_OPS[op_type](_safe_eval(node.operand, variables))

    # 비교 (a > b)
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left, variables)
        for op, right_node in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _ALLOWED_COMPARE_OPS:
                raise ValueError(f"허용되지 않는 비교 연산: {op_type.__name__}")
            right = _safe_eval(right_node, variables)
            if not _ALLOWED_COMPARE_OPS[op_type](left, right):
                return False
            left = right
        return True

    # 함수 호출 (abs, round, ...)
    if isinstance(node, ast.Call):
        func = _safe_eval(node.func, variables)
        if not callable(func):
            raise ValueError("함수 호출 불가")
        # 위치 인자만 (kwargs 차단)
        args = [_safe_eval(a, variables) for a in node.args]
        if node.keywords:
            raise ValueError("키워드 인자는 허용되지 않음")
        return func(*args)

    raise ValueError(f"허용되지 않는 노드 타입: {type(node).__name__}")


def safe_calculate(expression: str, variables: dict[str, float] | None = None) -> float:
    """문자열 수식을 *안전하게* 계산.

    Args:
        expression: 예 "(current - prev) / prev * 100"
        variables: 예 {"current": 20717, "prev": 19059}

    Returns:
        계산 결과 (float). 비교식이면 bool도 가능하지만 보통 float.

    Raises:
        ValueError, NameError, ZeroDivisionError 등 — 호출자가 catch.
    """
    variables = variables or {}
    tree = ast.parse(expression.strip(), mode="eval")
    result = _safe_eval(tree, variables)
    return result


# ── Tool 구현 ────────────────────────────────────────────────────

@register_tool(ActionType.CALCULATE)
class CalculateTool(ToolBase):
    """수식 계산 Tool.

    Agent가 *2개 이상 데이터 점*을 모은 후 계산이 필요할 때 호출.
    예: 증가율, 차이, 비율, 점유율 등.
    """

    name = ActionType.CALCULATE
    description = (
        "수치 계산. 모은 데이터 점들로 증가율/차이/비율을 계산할 때 사용. "
        "예: (current - prev) / prev * 100. "
        "허용 연산: + - * / % ** , 함수: abs, round, min, max, sqrt, log, log10."
    )
    input_schema = {
        "expression": "수식 문자열. 예: '(current - prev) / prev * 100'",
        "variables": "변수 dict. 예: {'current': 20717, 'prev': 19059}",
    }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        expression = input_data.get("expression", "").strip()
        variables = input_data.get("variables") or {}

        if not expression:
            return ToolResult(
                output={},
                summary="실패: expression 비어있음",
                success=False,
                error="expression이 비어있습니다.",
            )

        # variables의 모든 값이 숫자인지 검증
        try:
            variables = {k: float(v) for k, v in variables.items()}
        except (TypeError, ValueError) as e:
            return ToolResult(
                output={},
                summary=f"실패: variables 변환 오류 — {e}",
                success=False,
                error=f"variables의 값은 모두 숫자여야 합니다: {e}",
            )

        try:
            result = safe_calculate(expression, variables)
        except Exception as e:
            logger.info(f"[calculate] 실패: {expression!r} | {e}")
            return ToolResult(
                output={"expression": expression, "variables": variables},
                summary=f"계산 실패: {expression} → {type(e).__name__}: {e}",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        # 성공
        var_desc = ", ".join(f"{k}={v}" for k, v in variables.items()) if variables else "(no vars)"
        summary = f"계산: {expression} | {var_desc} = {result}"

        # bool인 경우 비교 결과
        if isinstance(result, bool):
            result_value: Any = result
        else:
            result_value = float(result)

        return ToolResult(
            output={
                "expression": expression,
                "variables": variables,
                "result": result_value,
            },
            summary=summary,
            success=True,
        )
