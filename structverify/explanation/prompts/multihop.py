MULTIHOP_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: {verdict_label} — 멀티홉 검증으로 판정했습니다.]
주장: "{claim_text}"
주장한 비율/배수: {claimed_ratio}배
계산된 비율/배수: {computed_ratio}배
근거: 원천 수치 {largest_value} / {smallest_value} = {computed_ratio}배
신뢰도: {confidence:.0%}

[작성 규칙]
- 이 주장은 KOSIS에서 직접 찾을 수 없는 "파생 주장"(비율/배수)입니다
- 대신 같은 지표의 원천 수치 2개를 KOSIS에서 찾아 비율을 직접 계산했습니다
- 2~3문장으로, 어떻게 계산했는지 설명: "원천 수치 {largest_value}와 {smallest_value}를 비교하면 약 {computed_ratio}배"
- 위에 적힌 수치만 사용하세요. 새 수치를 만들지 마세요."""
