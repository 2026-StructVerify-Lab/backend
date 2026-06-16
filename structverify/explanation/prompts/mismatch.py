MISMATCH_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: 불일치 (MISMATCH) — 기사 수치와 공식 수치가 다릅니다.]
주장: "{claim_text}"
기사 수치: {claimed_value} {unit}
공식 수치: {official_value} {unit}
차이: {diff} {unit} ({diff_pct:.1f}%)
불일치 유형: {mismatch_reason}
신뢰도: {confidence:.0%}
근거 통계: {stat_source}
출처: {provenance}

[작성 규칙]
- 3~4문장으로 작성
- 기사 수치({claimed_value})와 공식 수치({official_value})를 반드시 정확히 인용
- 위에 적힌 수치만 사용하세요. 새로운 수치를 만들어내지 마세요.
- 불일치 유형({mismatch_reason})에 맞는 원인 설명 포함
- KOSIS 출처 포함"""
