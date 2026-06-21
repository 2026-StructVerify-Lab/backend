"""detection/prompts/domain.py — Step 3 도메인 분류 프롬프트.

domain_classifier.py에서 분리 (문자열 move-only, 동작 변경 없음).

[김예슬 - 2026-04-22] DOMAIN_CLASSIFY_PROMPT few-shot 예시
[김예슬 - 2026-04-23] 기존 도메인 목록 동적 주입 방식
"""
from __future__ import annotations


DOMAIN_CLASSIFY_PROMPT = """당신은 한국 통계/뉴스 도메인 분류 전문가입니다.
아래 문서를 읽고, 가장 적합한 도메인을 선택하거나 새로 생성하세요.

[현재 등록된 도메인 목록]
{domain_list}

[도메인 선택 규칙]
1. 위 목록에서 문서 내용과 가장 잘 맞는 도메인이 있으면 그 도메인을 선택하세요.
2. 목록에 적합한 도메인이 없을 때만 새 도메인을 만드세요.
   - 영어 소문자와 언더스코어(_)만 사용 (예: real_estate, it_industry)
   - 새 도메인 설명은 한국어로 간략히 작성
3. 분류가 모호하거나 복합 도메인이면 "general"을 선택하세요.

[예시]
문서: "통계청에 따르면 지난해 농가 인구는 216만 명으로 전년 대비 3.2% 감소했다."
→ {{"domain": "agriculture", "description": "농림수산식품 (농가, 경작면적, 수확량, 축산, 어업)", "is_new": false, "confidence": 0.95, "reason": "농가 인구 통계"}}

문서: "수도권 아파트 평균 매매가가 8억을 돌파하며 역대 최고치를 기록했다."
→ {{"domain": "real_estate", "description": "부동산 (아파트, 매매가, 전세, 분양)", "is_new": true, "confidence": 0.93, "reason": "부동산 가격 통계로 기존 목록에 없음"}}

[분류할 문서]
{text_preview}

JSON으로만 답하세요:
{{
  "domain": "도메인명",
  "description": "도메인 한국어 설명",
  "is_new": true 또는 false,
  "confidence": 0.0~1.0,
  "reason": "한 줄 근거"
}}"""
