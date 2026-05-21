"""structverify.agent.reflect — Reflect Agent (Phase E).

ReAct 패턴의 *Reflect* 단계:
  매 iter 시작 시 LLM이 last_observation 보고 *다음 action 동적 결정*.

Phase D의 deterministic plan 따르기 → Phase E의 LLM-driven 결정.
룰베이스 (_select_best_row, _infer_claim_type 등)는 *fallback only*.

사용:
    from structverify.agent.reflect import ReflectAgent

    reflect = ReflectAgent(llm_call=my_llm_call, config={...})
    verdict = await agent_loop(
        plan=plan, claim=claim, workspace=ws, datasources=ds,
        reflect_fn=reflect,  # callable, agent_loop이 매 iter 호출
        loop_config=LoopConfig(mode="reflect"),
    )
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from structverify.utils.logger import get_logger
from .schemas import ActionType, Observation, Plan, ReflectDecision, VerdictType
from .prompts.reflect_prompts import build_reflect_prompt

logger = get_logger(__name__)


# ── JSON 추출 (planner와 동일 패턴) ──────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

# JSON 표준엔 주석 없는데 LLM이 자주 박음 — 파싱 전 제거
_LINE_COMMENT_RE = re.compile(r"//[^\n\r]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
# trailing comma도 흔함
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _clean_json_text(text: str) -> str:
    """LLM 응답에서 JSON 표준 위반 흔한 패턴 제거."""
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _extract_json(text: str) -> dict | None:
    """LLM 응답에서 JSON object 추출. fenced ``` 블록 우선, 그 다음 첫 { ... }."""
    if not text:
        return None

    # 1. fenced code block
    m = _JSON_FENCE_RE.search(text)
    if m:
        body = _clean_json_text(m.group(1).strip())
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

    # 2. 첫 { ... } 매칭
    m = _JSON_OBJ_RE.search(text)
    if m:
        body = _clean_json_text(m.group(0))
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

    # 3. 전체 text
    try:
        return json.loads(_clean_json_text(text.strip()))
    except json.JSONDecodeError:
        return None


def _parse_reflect_decision(response_text: str) -> ReflectDecision | None:
    """LLM 응답 → ReflectDecision. 실패 시 None (호출자가 fallback)."""
    data = _extract_json(response_text)
    if not data or not isinstance(data, dict):
        logger.warning(f"[reflect] JSON 추출 실패. 응답 일부: {response_text[:300]!r}")
        return None

    # action 파싱
    raw_action = (data.get("action") or "").strip().lower()
    action_map = {
        "catalog_search": ActionType.CATALOG_SEARCH,
        "explore_catalog": ActionType.EXPLORE_CATALOG,
        "fetch_evidence": ActionType.FETCH_EVIDENCE,
        "read_original": ActionType.READ_ORIGINAL,
        "calculate": ActionType.CALCULATE,
        "finish": ActionType.FINISH,
    }
    action = action_map.get(raw_action)
    if action is None:
        logger.warning(f"[reflect] 알 수 없는 action: {raw_action!r}")
        return None

    # input
    inp = data.get("input") or {}
    if not isinstance(inp, dict):
        logger.warning(f"[reflect] input이 dict 아님: {type(inp).__name__}")
        inp = {}

    # confidence
    try:
        conf = float(data.get("confidence_so_far", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    # proposed_verdict (finish action에서만 의미)
    raw_verdict = data.get("proposed_verdict")
    verdict_obj: VerdictType | None = None
    if raw_verdict:
        try:
            verdict_obj = VerdictType(str(raw_verdict).strip().lower())
        except ValueError:
            logger.debug(f"[reflect] proposed_verdict 파싱 실패: {raw_verdict!r}")

    return ReflectDecision(
        thought=str(data.get("thought") or "").strip(),
        action=action,
        input=inp,
        confidence_so_far=conf,
        proposed_verdict=verdict_obj,
        proposed_explanation=(data.get("proposed_explanation") or None),
    )


# ── ReflectAgent ─────────────────────────────────────────────────

@dataclass
class ReflectConfig:
    model_tier: str = "light"
    """LLM 모델 선택. heavy는 비용 큼."""
    temperature: float = 0.2
    """낮을수록 일관적. 0.2~0.3 권장."""
    max_retries: int = 1
    """JSON 파싱 실패 시 retry 횟수."""


class ReflectAgent:
    """LLM 호출 → ReflectDecision 반환.

    agent_loop의 reflect_fn 시그니처에 맞춤:
        async __call__(plan, memory_text, last_observation, iter_num) -> ReflectDecision | None

    실패 시 None 반환 → loop이 deterministic fallback (plan의 다음 step).
    """

    def __init__(
        self,
        llm_call: Callable[..., Awaitable[str]],
        claim: Any,
        config: ReflectConfig | None = None,
        max_iterations: int = 10,
    ):
        """
        Args:
            llm_call: async function. signature: `async (prompt: str) -> str`.
                      (보통 lambda로 model_tier/temperature를 binding)
            claim: 현재 claim. schema 정보 접근용.
            config: ReflectConfig
            max_iterations: prompt 안에 표시할 max iter
        """
        self.llm_call = llm_call
        self.claim = claim
        self.config = config or ReflectConfig()
        self.max_iterations = max_iterations
        self._call_count = 0

    async def __call__(
        self,
        plan: Plan,
        memory_text: str,
        last_observation: Observation | None,
        iter_num: int,
    ) -> ReflectDecision | None:
        """매 iter 호출됨. ReflectDecision 또는 None (fallback)."""
        self._call_count += 1

        prompt = build_reflect_prompt(
            claim=self.claim,
            plan=plan,
            memory_text=memory_text,
            last_observation=last_observation,
            iter_num=iter_num,
            max_iterations=self.max_iterations,
        )

        for attempt in range(1, self.config.max_retries + 2):  # 최초 1회 + retry
            try:
                response = await self.llm_call(prompt)
            except Exception as e:
                logger.warning(
                    f"[reflect] LLM 호출 실패 (iter={iter_num}, attempt={attempt}): "
                    f"{type(e).__name__}: {e}"
                )
                continue

            decision = _parse_reflect_decision(response)
            if decision is not None:
                logger.info(
                    f"[reflect] iter {iter_num}: action={decision.action.value} "
                    f"thought={decision.thought[:120]!r} "
                    f"confidence={decision.confidence_so_far:.2f}"
                )
                return decision

            logger.warning(
                f"[reflect] iter {iter_num} attempt {attempt}: 파싱 실패. "
                f"응답 일부: {response[:200]!r}"
            )

        # 모든 시도 실패 → None (loop이 deterministic fallback)
        logger.warning(
            f"[reflect] iter {iter_num}: 모든 시도 실패, deterministic fallback"
        )
        return None