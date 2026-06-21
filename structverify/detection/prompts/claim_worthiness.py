"""detection/prompts/claim_worthiness.py — Step 4 check-worthiness 프롬프트.

claim_detector.py에서 분리 (문자열 move-only, 동작 변경 없음).

[박재윤 - 2026-05-14] CHECK_WORTHY_PROMPT 개선
[박재윤 - 2026-05-18] 검증 가능 기준 보강
"""
from __future__ import annotations


# TODO [김예슬]: 프롬프트 튜닝
#   - domain-packs/{domain}/prompts.yaml에서 도메인별 few-shot 예시 로드
#   - positive 예시: 공식 통계로 검증 가능한 수치 주장 2~3개
#   - negative 예시: 의견/감상/단순 이벤트 일정 2~3개
#   - claim_type 분류 기준 명확화 (increase/decrease/scale/comparison/forecast)
CHECK_WORTHY_PROMPT = """팩트체크 전문가로서 아래 문장이 공식 통계로 검증 가능한 수치 기반 주장인지 판별하세요.

[검증 가능 기준]
1. 정부/공공기관이 발표한 구체적 수치가 포함된 사실 주장
   (변동률, 상승률, 하락률, 비율, 절대값, 증감폭 등 모두 포함)
2. 단순 일정/발언 소개/감상이 아닌 사실 주장
3. 수치가 *과거 또는 현재* 실측값 (예보/예상/전망/목표 아님)

[검증 불가 기준 — is_check_worthy=false]
- 예보/예상/전망/목표 수치: "예상 강수량 20mm", "목표 성장률 3%"
- 순위 표현만: "34년 만에 최대", "역대 최고"
- 단순 발언/의견: "전문가는 ~라고 말했다"
- 외국 기관 발표 수치 (KOSIS 검증 불가): "미국 연준이 금리를 0.25% 올렸다"

[검증 가능 예시]
✓ "2024년 4월 출생아 수는 2만 171명이다" → true
✓ "고용률이 전년 대비 1.2% 상승했다" → true
✓ "서울 표준주택 공시가격 상승률은 6.8%다" → true
✓ "동작구 공시가 상승률은 10.6%로 가장 높다" → true
✓ "전국 표준단독주택 공시가격 상승률은 4.5%다" → true
✗ "올해 강수량이 20mm로 예상된다" → false
✗ "출생아 수가 34년 만에 최대를 기록했다" → false

문장: "{sentence}"

중요:
- is_check_worthy=true이면 score는 반드시 0.5 이상
- is_check_worthy=false이면 score는 반드시 0.5 미만
- JSON만 출력. 설명 금지.

JSON:
{{"is_check_worthy": false, "score": 0.0, "claim_type": null}}
"""
