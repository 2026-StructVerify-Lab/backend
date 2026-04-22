"""
detection/candidate_scorer.py — 문장 단위 검증 후보 점수화

설계 원칙
- regex만으로 후보를 결정하지 않는다.
- surface signal + teacher LLM + weak supervision 규칙을 결합할 수 있는 인터페이스를 제공한다.
- 현재 버전은 "teacher LLM + heuristic fallback" 구조로 작성.
- 이후 작은 classifier를 붙일 때 이 파일만 교체하면 된다.

출력
- candidate_score: 0~1
- candidate_label: bool
- candidate_source: 점수 출처
- candidate_signals: 분석용 signal
"""
from __future__ import annotations

import re
from typing import Any

from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

CANDIDATE_PROMPT = """당신은 수치 기반 팩트체크 시스템의 1차 후보 탐지기입니다.
아래 문장이 "공식 통계나 구조화된 데이터로 검증할 만한 후보 문장"인지 판단하세요.

판단 기준:
1. 수치/비율/규모/시점/대상 중 일부가 드러나는가?
2. 의견/감상/단순 이벤트 일정이 아니라 검증 가능한 사실 주장인가?
3. 공식 통계 또는 공공 데이터와 연결될 가능성이 있는가?

문장: "{sentence}"

JSON으로만 답하세요:
{{
  "candidate_score": 0.0,
  "candidate_label": false,
  "reason": "짧은 근거",
  "signals": {{
    "has_quantity": false,
    "has_time_expr": false,
    "has_population": false,
    "has_comparison_expr": false
  }}
}}
"""

TIME_PATTERN = re.compile(r"\d{4}년|\d+월|\d+분기|전년|지난해|올해")
COMPARISON_PATTERN = re.compile(r"증가|감소|상승|하락|올랐다|내렸다|대비|비율|점유율|이상|이하|안팎")
POPULATION_PATTERN = re.compile(r"국내|전국|가구|가계|농가|학생|청년|고령자|근로자|기업|미국|일본|유럽|한국")
NUMBER_PATTERN = re.compile(r"\d")


async def score_candidate(
    sentence: str,
    config: dict | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[float, bool, str, dict[str, Any]]:
    """
    문장 후보 점수 계산.

    현재 로직
    1) teacher LLM 시도
    2) 실패 시 heuristic fallback
    """
    config = config or {}
    cd_cfg = config.get("candidate_detection", {})
    use_llm = cd_cfg.get("teacher_llm_fallback", True)
    threshold = float(cd_cfg.get("threshold", 0.65))

    if use_llm:
        try:
            llm = LLMClient(config=config.get("llm", {}))
            result = await llm.generate_json(
                prompt=CANDIDATE_PROMPT.format(sentence=sentence),
                system_prompt="팩트체크 candidate detector. JSON으로만 답하세요.",
            )
            score = float(result.get("candidate_score", 0.0))
            label = bool(result.get("candidate_label", score >= threshold))
            signals = result.get("signals", {}) or {}
            signals["reason"] = result.get("reason")
            return score, label, "teacher_llm", signals
        except Exception as e:
            logger.warning(f"candidate LLM 판별 실패 — heuristic fallback 사용: {e}")

    return _score_candidate_heuristic(sentence, threshold=threshold)


def _score_candidate_heuristic(
    sentence: str,
    threshold: float = 0.65,
) -> tuple[float, bool, str, dict[str, Any]]:
    """
    최소한의 fallback heuristic.
    논문 실험에서 baseline으로도 사용 가능.
    """
    has_quantity = bool(NUMBER_PATTERN.search(sentence))
    has_time_expr = bool(TIME_PATTERN.search(sentence))
    has_population = bool(POPULATION_PATTERN.search(sentence))
    has_comparison_expr = bool(COMPARISON_PATTERN.search(sentence))

    score = 0.0
    if has_quantity:
        score += 0.35
    if has_time_expr:
        score += 0.20
    if has_population:
        score += 0.20
    if has_comparison_expr:
        score += 0.25

    score = min(score, 1.0)
    label = score >= threshold

    signals = {
        "has_quantity": has_quantity,
        "has_time_expr": has_time_expr,
        "has_population": has_population,
        "has_comparison_expr": has_comparison_expr,
        "reason": "heuristic_fallback",
    }
    return score, label, "heuristic_fallback", signals