"""detection/schema/validate.py — schema induction 수치·source_phrase 검증.

schema_inductor.py에서 분리 (로직 move-only, 동작 변경 없음).

[v6.14] context leak 방지 — _source_phrase_in_claim
[v6.14 E] value 환산 정확성 — _verify_and_correct_value
[박재윤 2026-05-18] _extract_numbers_from_text "N만 NNNN" 패턴
"""
from __future__ import annotations

import re
from typing import Any

from structverify.core.schemas import ClaimSchema


# ── [v6.14 E fix] value 환산 정확성 검증 ───────────────────────────────

def _verify_and_correct_value(
    value: float | None,
    source_phrase: str,
) -> tuple[float | None, bool]:
    """
    LLM이 보낸 value의 환산 정확성을 source_phrase로부터 검증.

    "2만 171" → 21710 같은 환산 오류를 잡아냄:
    - source_phrase에서 우리 코드로 수치 추출 (한글 단위 포함)
    - 그 수치 집합 중 value와 *가장 가까운 값* 찾음
    - 차이가 0.5% 이상이면 → 그 가까운 값으로 교체

    Returns:
        (corrected_value, was_corrected_flag)
    """
    if value is None or not source_phrase:
        return value, False

    sp_numbers = _extract_numbers_from_text(source_phrase)
    if not sp_numbers:
        return value, False  # 환산 불가능한 표현 → 그대로

    # value에 가장 가까운 수
    closest = min(sp_numbers, key=lambda n: abs(n - value))

    # 차이 작으면 OK
    if abs(closest - value) < 0.001:
        return value, False
    if value != 0 and abs(closest - value) / abs(value) < 0.005:
        return value, False

    # 차이 큼 — closest로 교체
    return float(closest), True


# ── [v6.14] Context leak 방지를 위한 검증 헬퍼 ───────────────────────────────

def _source_phrase_in_claim(phrase: str, claim_text: str) -> bool:
    """
    LLM이 제공한 source_phrase가 claim_text에 등장하는지 검증.

    [v6.15] 3단계 비교:
      1) 직접 substring
      2) 공백 제거 후 비교
      3) 숫자 기준 비교 — source_phrase의 모든 숫자가 claim_text에 있으면 통과
         ("6.8%" vs "6.8%↑" 처럼 기호 차이로 1·2단계가 실패하는 경우 대응)
    """
    if not phrase or not claim_text:
        return False
    # 1) 직접 substring
    if phrase in claim_text:
        return True
    # 2) 공백 제거 후 비교 (예: "1만 7921 건" vs "1만 7921건")
    phrase_no_space = re.sub(r"\s+", "", phrase)
    claim_no_space = re.sub(r"\s+", "", claim_text)
    if phrase_no_space in claim_no_space:
        return True
    # 3) [v6.15] 숫자 기준 비교 — 기호(↑↓%、 등) 차이 흡수
    #    source_phrase의 숫자들이 모두 claim_text 안에 있으면 leak 아님
    phrase_nums = re.findall(r"\d+\.?\d*", phrase)
    if phrase_nums:
        claim_nums = set(re.findall(r"\d+\.?\d*", claim_text))
        if all(n in claim_nums for n in phrase_nums):
            return True
    return False


def _value_in_claim_text(value: float, claim_text: str) -> bool:
    """
    source_phrase가 누락된 경우 fallback. value의 환산 전 표기가 문장에 있는지 검증.

    예: value=20171 → "2만 171", "20171", "20,171" 등 매칭 시도.
    """
    if value is None:
        return True
    if not claim_text:
        return False

    # 텍스트에서 모든 수치(한글 단위 포함) 추출 → 집합 만들고 value와 매칭
    numbers = _extract_numbers_from_text(claim_text)
    for n in numbers:
        if abs(n - value) < 0.001:
            return True
        if value != 0 and abs(n - value) / abs(value) < 0.005:
            return True
    return False


def _extract_numbers_from_text(text: str) -> set[float]:
    """
    텍스트에서 한글 단위 포함 모든 수치 추출.

    - "2만 171" → 20171
    - "1만 9059" → 19059
    - "6.7" → 6.7
    - "0.76" → 0.76
    - "238,317" → 238317
    """
    numbers: set[float] = set()

    # [박재윤 - 2026-05-18] "N만 N천" 복합 패턴 (앞 패턴보다 먼저 실행 필요)
    # "2869만 3000명" → 28693000
    # "159만 명" 같은 경우와 구분: 뒤 숫자가 1000 단위인 경우
    for m in re.finditer(r"(\d+)\s*만\s*(\d{4})", text):
        n = int(m.group(1)) * 10000 + int(m.group(2))
        numbers.add(float(n))

    # 1) 한글 단위 — "N만 M" 또는 "N만"
    for m in re.finditer(r"(\d+)\s*만\s*(\d+)?", text):
        n = int(m.group(1)) * 10000
        if m.group(2):
            n += int(m.group(2))
        numbers.add(float(n))


    # 2) 한글 단위 — "N억 M" 또는 "N억"
    for m in re.finditer(r"(\d+)\s*억\s*(\d+)?", text):
        n = int(m.group(1)) * 100_000_000
        if m.group(2):
            n += int(m.group(2))
        numbers.add(float(n))

    # 3) 한글 단위 — "N천 M" (앞에 만/억 없을 때만)
    for m in re.finditer(r"(?<![만억\d])(\d+)\s*천\s*(\d+)?", text):
        n = int(m.group(1)) * 1000
        if m.group(2):
            n += int(m.group(2))
        numbers.add(float(n))
    
    # "N만 M천" 복합 패턴
    for m in re.finditer(r"(\d+)\s*만\s*(\d+)\s*천", text):
        n = int(m.group(1)) * 10000 + int(m.group(2)) * 1000
        numbers.add(float(n))

    # 4) 일반 숫자 (콤마 포함 정수 + 소수)
    for m in re.finditer(r"[\d,]+(?:\.\d+)?", text):
        s = m.group().replace(",", "")
        if not s or s in (".",):
            continue
        try:
            numbers.add(float(s))
        except ValueError:
            pass

    return numbers


def _validate_schema(schema: ClaimSchema) -> bool:
    """indicator 없으면 KOSIS 검색 불가 → 실패 처리."""
    if not schema.indicator or len(schema.indicator.strip()) < 2:
        return False
    return True


def _safe_float(v: Any) -> float | None:
    """다양한 수치 표현 → float 변환.

    LLM이 이미 한글 단위를 환산해줘야 하지만, 혹시 문자열로 넘어올 때를 위한 백업.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = re.sub(r"[%,약\s]", "", v.strip())
        match = re.search(r"-?[\d.]+", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None
