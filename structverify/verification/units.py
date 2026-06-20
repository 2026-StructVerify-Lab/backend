"""[리팩] 단위 변환·타입 비교 — verifier에서 분리 (로직 동일)"""


def normalize_value(value: float, kosis_unit: str) -> float:
    """
    KOSIS 단위 → 기본 단위 변환.
    [v3] 천명개월은 실제로 개월 단위 (KOSIS 단위명 오류) → 변환 안 함.
    """
    u = (kosis_unit or "").lower()
    if "천명개월" in u:
        return value
    if "천" in u:
        return value * 1_000
    if "백만" in u or "million" in u:
        return value * 1_000_000
    if "억" in u:
        return value * 100_000_000
    return value


def is_same_unit_type(
    claim_unit: str,
    kosis_unit: str,
    all_rows_empty: bool = False,
) -> bool:
    """
    단위 타입이 같은지 확인 (명 ↔ 개월 혼용 방지).
    [v3] 천명개월은 KOSIS 단위명 오류 → 통과.
    """
    c = (claim_unit or "").lower().strip()
    k = (kosis_unit or "").lower().strip()

    if not c:
        return True

    if not k:
        return all_rows_empty

    if "천명개월" in k:
        return True

    _TYPES = {
        "people": ["명", "인구", "가구", "세대", "person"],
        "time":   ["개월", "월", "month", "년", "일", "주"],
        "ratio":  ["%", "퍼센트", "percent", "율", "비율"],
        "money":  ["원", "won", "달러", "dollar", "usd"],
    }

    def _get(u: str) -> str:
        for t, kws in _TYPES.items():
            if any(kw in u for kw in kws):
                return t
        return "unknown"

    ct, kt = _get(c), _get(k)
    return (ct == "unknown" or kt == "unknown") or (ct == kt)
