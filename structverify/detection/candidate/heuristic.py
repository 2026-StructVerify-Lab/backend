"""detection/candidate/heuristic.py — candidate scoring heuristic fallback.

candidate_scorer.py에서 분리 (로직 move-only).

LLM 실패 시만 사용 — rule 기반 1차 후보 결정 용도 아님.
"""
from __future__ import annotations

import re
from typing import Any

# ── heuristic fallback 패턴 (LLM 실패 시만 사용) ────────────────────────
# 아래 패턴들은 LLM이 호출 불가능할 때만 사용하는 fallback입니다.
# 운영 환경에서는 LLM 판단이 우선입니다.
TIME_PATTERN = re.compile(r"\d{4}년|\d+월|\d+분기|전년|지난해|올해")
COMPARISON_PATTERN = re.compile(r"증가|감소|상승|하락|올랐다|내렸다|대비|비율|점유율|이상|이하|안팎")
POPULATION_PATTERN = re.compile(r"국내|전국|가구|가계|농가|학생|청년|고령자|근로자|기업|미국|일본|유럽|한국")
NUMBER_PATTERN = re.compile(r"\d")


def _score_candidate_heuristic(
    sentence: str,
    threshold: float = 0.65,
) -> tuple[float, bool, str, dict[str, Any]]:
    """
    최소한의 fallback heuristic.

    TODO [김예슬]: 논문 실험 baseline으로도 활용 가능
      - 이 함수의 성능(F1, precision, recall)을 측정하고
        teacher LLM 및 fine-tuned 모델과 비교

    주의: 이 heuristic은 LLM 호출 실패 시만 사용합니다.
    Rule 기반으로 검증 후보를 결정하는 용도로 사용하지 마세요.
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
