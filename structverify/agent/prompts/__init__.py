"""structverify.agent.prompts — Agent 시스템에서 사용하는 LLM 프롬프트 모음.

각 모듈:
  - planner_prompts: Plan Agent (Phase C) — claim → Plan JSON
  - reflect_prompts: Reflect Agent (Phase D) — 현 상태 → 다음 ActionType
  - 향후: explainer_prompts (기존 explainer.py 통합), verifier_prompts (필요 시)

분리 이유:
  - 프롬프트는 *자주 수정됨* — 로직 코드와 섞이면 PR 리뷰가 어렵다
  - 한국어/영어 버전 분기 시 깔끔
  - 다른 LLM 시도 시 (HCX → Claude → GPT) prompt만 갈아끼우면 됨
"""
from .planner_prompts import PLAN_PROMPT_TEMPLATE, build_plan_prompt

__all__ = ["PLAN_PROMPT_TEMPLATE", "build_plan_prompt"]
