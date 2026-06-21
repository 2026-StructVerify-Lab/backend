"""[리팩] 단위 변환·타입 비교 — verifier에서 분리 (로직 동일)"""


def normalize_value(value: float, kosis_unit: str) -> float:
    """
    KOSIS 단위 → 기본 단위 변환.
    [v3] 천명개월은 실제로 개월 단위 (KOSIS 단위명 오류) → 변환 안 함.
    """
    u = (kosis_unit or "").lower()
    # 천명개월은 KOSIS 단위명 오류 — 실제로는 개월 단위
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
    [v6.14] 비대칭 처리:
      - claim_unit 비어있으면 → True (claim 측 정보 부족, 책임은 claim에)
      - kosis_unit 비어있으면 → False (KOSIS row 단위 없으면 안전 차단)
    [v6.14 G fix] all_rows_empty: 표 전체 unit 빈 칸이면 지표명-단위 일체형 표로 통과.
    """
    c = (claim_unit or "").lower().strip()
    k = (kosis_unit or "").lower().strip()

    # [v6.14] claim 측 단위가 없는 경우 → 통과 (claim 책임)
    if not c:
        return True

    # [v6.14] KOSIS row 단위가 없는 경우
    if not k:
        return all_rows_empty

    # 천명개월은 KOSIS 단위명 오류 — 비교 자체를 통과
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
