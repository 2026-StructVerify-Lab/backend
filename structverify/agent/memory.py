"""
structverify.agent.memory — Agent의 멀티턴 메모리.

Memory는 *평문 markdown*으로 저장되며, agent가 매 iteration마다 *전체 읽고
새 내용 추가*한다. memory.md가 너무 길어지면 *요약/압축* (Phase D+에서).

저장 자체는 Workspace.append_memory / read_memory를 통하지만, 이 모듈은:
  - 일관된 포맷으로 새 항목 추가 (append_iteration, append_plan_summary 등)
  - LLM에 넘기기 좋은 형태로 read (read_for_llm — 길이 제한 적용)
  - 이미 시도한 action 추적 (중복 방지)

Phase A에서는 *저장/읽기 헬퍼*만. 요약(summarize_memory)은 Phase D+.
"""
from __future__ import annotations

import json
from structverify.utils.logger import get_logger
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .workspace import Workspace

logger = get_logger(__name__)


# ── Append 헬퍼 (일관된 포맷) ─────────────────────────────────────

def append_plan_summary(ws: Workspace, claim_id: str | UUID, plan: dict) -> None:
    """Plan Agent 결과를 memory 시작 부분에 기록.

    Plan Agent (Phase C)가 호출.
    """
    required = plan.get("required_data", [])
    required_lines = []
    for d in required:
        if isinstance(d, dict):
            ind = d.get("indicator", "?")
            t = d.get("time", "?")
            pop = d.get("population", "")
            line = f"  - {ind} (time={t}" + (f", pop={pop}" if pop else "") + ")"
        else:
            line = f"  - {d}"
        required_lines.append(line)

    text = (
        "## Initial Plan\n"
        f"Claim type: {plan.get('claim_type', 'unknown')}\n"
        f"Required data:\n" + "\n".join(required_lines) + "\n"
        f"Formula: {plan.get('calculation_formula') or '(none — direct comparison)'}\n"
        f"Fallback: use_original_text={plan.get('fallback', {}).get('use_original_text', False)}\n"
    )
    ws.append_memory(claim_id, text)


def append_iteration(
    ws: Workspace,
    claim_id: str | UUID,
    iteration_num: int,
    action: str,
    action_input: dict,
    observation_summary: str,
    reflection: str | None = None,
    success: bool = True,
) -> None:
    """
    한 iteration 결과를 memory.md에 추가.

    포맷:
        ## Iteration {n} — {action}
        Input: {input}
        Result: {summary}
        Reflection: {reflection}     (있으면)
        Status: success | failed
    """
    status_marker = "✓" if success else "✗"
    lines = [
        f"## Iteration {iteration_num} — {action} {status_marker}",
        f"Input: {_format_input(action_input)}",
        f"Result: {observation_summary}",
    ]
    if reflection:
        lines.append(f"Reflection: {reflection}")
    lines.append("")  # 빈 줄

    ws.append_memory(claim_id, "\n".join(lines))


def append_final(
    ws: Workspace,
    claim_id: str | UUID,
    verdict: str,
    confidence: float,
    reason: str,
    iterations_used: int,
) -> None:
    """최종 판정을 memory 끝에 기록."""
    text = (
        "## Final Verdict\n"
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Iterations used: {iterations_used}\n"
        f"Reason: {reason}\n"
    )
    ws.append_memory(claim_id, text)


# ── LLM-friendly read ───────────────────────────────────────────

def read_for_llm(
    ws: Workspace,
    claim_id: str | UUID,
    max_chars: int = 50_000,
) -> str:
    """
    LLM에 넘길 memory 텍스트. 너무 길면 *앞부분(헤더 + Plan)*은 보존하고
    *중간 iteration 일부*를 잘라낸다. 최신 iteration이 가장 중요.

    Phase D에서 Reflect Agent가 호출.

    Args:
        max_chars: 글자 수 상한. 보통 LLM token limit의 절반 정도.
    """
    text = ws.read_memory(claim_id)
    if len(text) <= max_chars:
        return text

    # 헤더(== ~ ## Initial Plan)를 보존, 그 뒤 일부 잘라내고 최신 부분 유지
    plan_header_marker = "## Initial Plan"
    first_iter_marker = "## Iteration 1"

    header_end = text.find(first_iter_marker)
    if header_end == -1:
        # Plan이 없거나 모름 — 그냥 뒤쪽 max_chars만
        return text[-max_chars:]

    header = text[:header_end]
    budget_for_recent = max_chars - len(header) - 300  # 안내 텍스트 여유
    if budget_for_recent <= 0:
        # 헤더가 이미 너무 큼
        return text[-max_chars:]

    recent = text[-budget_for_recent:]
    note = "\n\n[... 중간 iteration 일부 생략됨 — 최신만 표시 ...]\n\n"
    return header + note + recent


# ── 이미 시도한 action 추적 (중복 방지) ─────────────────────────

def get_attempted_actions(
    ws: Workspace,
    claim_id: str | UUID,
) -> list[dict[str, Any]]:
    """
    이미 시도한 (action, input) 목록.

    Reflect Agent (Phase D)가 *같은 검색어 반복 시도 방지*에 사용.
    log.jsonl에서 읽음.
    """
    log = ws.read_log(claim_id)
    attempts: list[dict[str, Any]] = []
    for entry in log:
        action = entry.get("action")
        inp = entry.get("input", {})
        if action:
            attempts.append({"action": action, "input": inp})
    return attempts


def has_attempted(
    ws: Workspace,
    claim_id: str | UUID,
    action: str,
    input_data: dict,
) -> bool:
    """동일 action+input이 이미 시도됐는지."""
    attempts = get_attempted_actions(ws, claim_id)
    for a in attempts:
        if a["action"] == action and a["input"] == input_data:
            return True
    return False


# ── 내부 헬퍼 ────────────────────────────────────────────────────

def _format_input(inp: dict | None) -> str:
    """간결한 input 표현."""
    if not inp:
        return "(none)"
    if len(inp) == 1:
        k, v = next(iter(inp.items()))
        return f"{k}={v!r}"
    parts = []
    for k, v in inp.items():
        # 값이 너무 길면 잘라냄
        v_str = repr(v)
        if len(v_str) > 80:
            v_str = v_str[:77] + "..."
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


# ── Phase D+에서 추가될 자리 ──────────────────────────────────────
#
# def summarize_memory(ws, claim_id, llm) -> str:
#     """Memory가 너무 길면 LLM으로 요약 (Phase D+)."""
#     ...
