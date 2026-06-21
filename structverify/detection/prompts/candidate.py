"""detection/prompts/candidate.py — Step 4 candidate scoring 프롬프트.

candidate_scorer.py에서 분리 (문자열 move-only, 동작 변경 없음).
"""
from __future__ import annotations


# TODO [김예슬]: 프롬프트 튜닝 — domain-packs의 few-shot 예시 주입
#   - 도메인별 positive/negative 예시 2~3개씩 추가
#   - "공식 통계와 연결 가능" 기준을 예시로 명확히 제시
CANDIDATE_PROMPT = """당신은 수치 기반 팩트체크 시스템의 1차 후보 탐지기입니다.
아래 문장이 "공식 통계나 구조화된 데이터로 검증할 만한 후보 문장"인지 판단하세요.

판단 기준:
1. 수치/비율/규모/시점/대상 중 일부가 드러나는가?
2. 의견/감상/단순 이벤트 일정이 아니라 검증 가능한 사실 주장인가?
3. 공식 통계 또는 공공 데이터와 연결될 가능성이 있는가?

문장: "{sentence}"

중요:
- candidate_label이 true이면 candidate_score는 반드시 0.5 이상이어야 합니다.
- candidate_label이 false이면 candidate_score는 반드시 0.5 미만이어야 합니다.
- JSON 앞뒤에 설명 문장을 절대 붙이지 마세요.

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
