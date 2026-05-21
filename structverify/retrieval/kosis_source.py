"""structverify.retrieval.kosis_source — KOSIS DataSource adapter (wire-completed).

사용자 코드의 *KOSISConnector*를 *BaseDataSource* 인터페이스로 wrap.

핵심 설계:
  - KOSISConnector 인스턴스를 *주입* (runtime_agent.self.kosis 재사용)
  - 새 connector를 만들지 않음 — *기존 인스턴스 그대로 사용*
  - 시그니처: KOSISConnector.search(ConnectorQuery) + .fetch(stat_id, params)

Phase D 개선사항:
  1. claim.schema → time_period 자동 KOSIS 파라미터 변환
     ("2025-04" → prdSe="M" startPrdDe="202504" endPrdDe="202504")
  2. ★ rows 안에서 indicator + time_period 매칭 row 직접 선택
     (connector가 drows[0]만 official_value로 만들어 통합값을 받는 문제 해결.
      DT_1B8000G 같은 출생/사망/혼인/이혼 통합표에서 정확한 row 추출)
"""
from __future__ import annotations

import re
from typing import Any

from .base import BaseDataSource, CatalogCandidate, EvidenceData
from .registry import register_datasource
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── time_period 파싱 helper ──────────────────────────────────────

def _parse_time_period(tp: str) -> tuple[str, str, str]:
    """time_period 문자열 → (prdSe, startPrdDe, endPrdDe) KOSIS API 파라미터.

    KOSIS API spec:
      - prdSe: 수록주기. "M"(월), "Q"(분기), "Y"(연간), "IR"(부정기)
      - startPrdDe / endPrdDe: 시점 (prdSe에 따라 형식 다름)
        · M: YYYYMM (예: "202504")
        · Q: YYYY0Q (예: "20251" = 2025년 1분기)
        · Y: YYYY    (예: "2025")

    예시:
      "2025-04" → ("M", "202504", "202504")  — 월별
      "2025-Q1" → ("Q", "20251",  "20251")    — 분기
      "2025"    → ("Y", "2025",   "2025")     — 연간
      ""        → ("Y", "",       "")         — fallback
    """
    if not tp:
        return ("Y", "", "")
    tp = str(tp).strip()

    # YYYY-MM, YYYY.MM, YYYY/MM
    m = re.match(r"^(\d{4})[-./](\d{1,2})$", tp)
    if m:
        year, month = m.group(1), m.group(2).zfill(2)
        return ("M", f"{year}{month}", f"{year}{month}")

    # YYYY-Q1, YYYY.Q1, YYYY/Q1, YYYYQ1
    m = re.match(r"^(\d{4})[-./]?Q([1-4])$", tp, re.IGNORECASE)
    if m:
        year, q = m.group(1), m.group(2)
        return ("Q", f"{year}0{q}", f"{year}0{q}")

    # YYYY only
    m = re.match(r"^(\d{4})$", tp)
    if m:
        return ("Y", m.group(1), m.group(1))

    return ("Y", "", "")


def _normalize_prd_de(tp: str) -> str:
    """time_period를 KOSIS PRD_DE 포맷으로 변환.

    "2025-04" → "202504"
    "2025"    → "2025"
    "" / None → ""
    """
    if not tp:
        return ""
    tp = str(tp).strip()
    m = re.match(r"^(\d{4})[-./](\d{1,2})$", tp)
    if m:
        return f"{m.group(1)}{m.group(2).zfill(2)}"
    m = re.match(r"^(\d{4})[-./]?Q([1-4])$", tp, re.IGNORECASE)
    if m:
        return f"{m.group(1)}0{m.group(2)}"
    m = re.match(r"^(\d{4})$", tp)
    if m:
        return m.group(1)
    return tp.replace("-", "").replace(".", "").replace("/", "")


def _period_granularity(prd: str) -> str:
    """[v6.19] PRD_DE 문자열의 시점 단위를 판별.

    "202409" (6자리) → "month"
    "20243"  (5자리, 분기) → "quarter"
    "2024"   (4자리) → "year"
    그 외 → "unknown"

    claim 시점과 KOSIS 행 시점의 단위가 다른지(월 claim에 연값 매칭)
    가드하는 데 쓴다.
    """
    if not prd:
        return "unknown"
    p = str(prd).strip()
    if not p.isdigit():
        return "unknown"
    if len(p) == 6:
        return "month"
    if len(p) == 5:
        return "quarter"
    if len(p) == 4:
        return "year"
    return "unknown"


def _table_has_period_for(rows: list[dict], claim_period: str) -> bool:
    """[v6.20] 표(rows)에 claim 시점 단위의 데이터가 존재하는지 판별.

    도메인 독립적 규칙: claim이 월(YYYY-MM)인데 표의 모든 행이
    연 단위(PRD_SE='A' 또는 PRD_DE 4자리/없음)면, 그 표엔 월
    데이터가 없는 것 → fetch해도 가짜 매칭만 나오므로 일찍 거부한다.

    KOSIS 표마다 수록주기가 다르다 (DT_1YL9801=연, DT_1B8000G의
    일부 행=연만 응답 등). 표별 하드코딩 대신 행 데이터로 판별해
    어느 도메인 표에도 동작하게 한다.

    Returns:
        True  — claim 단위의 행이 하나라도 있음 (또는 판별 불가 → 통과)
        False — claim은 월/분기인데 표는 연 단위뿐 → 거부 권장
    """
    claim_gran = _period_granularity(_normalize_prd_de(claim_period or ""))
    if claim_gran not in ("month", "quarter"):
        return True  # 연 단위 claim — 가드 불필요
    if not rows:
        return True  # 행 없음 — 별도 처리
    saw_any_period = False
    for r in rows:
        prd = str(r.get("PRD_DE", "") or "").strip()
        prd_se = str(r.get("PRD_SE", "") or "").strip().upper()
        row_gran = _period_granularity(prd)
        if prd_se == "A":
            row_gran = "year"
        if row_gran != "unknown":
            saw_any_period = True
        # claim 단위(월/분기)와 호환되는 행을 하나라도 찾으면 OK
        if claim_gran == "month" and row_gran == "month":
            return True
        if claim_gran == "quarter" and row_gran in ("month", "quarter"):
            return True
    # 시점 정보가 있는 행은 봤지만 claim 단위와 맞는 게 하나도 없음 → 거부
    # 시점 정보가 아예 없는 표(saw_any_period=False)는 판별 불가 → 통과
    return not saw_any_period


def _normalize_korean(s: str) -> str:
    """KOSIS 한국어 비교용 정규화 — 공백/특수문자 제거, 소문자."""
    if not s:
        return ""
    s = str(s)
    # 공백, 중간점, 슬래시 등 표기 차이 흡수
    for ch in (" ", "\t", "\u00A0", "·", "ㆍ", "・", "/", "-", "_", "(", ")", "[", "]"):
        s = s.replace(ch, "")
    return s.strip().lower()


# ── [수정 v6.23] 국가 차원 인식 ───────────────────────────────────────
# [추가 이유] DT_2KAA207('합계출산율')처럼 표 이름엔 국가 표시가 없지만
#   실제 행은 국가별(세계/대한민국/일본/중국...)인 표가 있다. 표 이름만
#   보는 가드는 이런 표를 통과시키고, 행 선택은 첫 행('세계' 2.25)을
#   집어 한국 기사값(0.72)과 비교 → '불일치(거짓)'로 오판한다.
# [해결] 가져온 행 데이터의 카테고리 컬럼을 보고 '국가 차원'인지
#   판별한다. 국가 차원이면 반드시 '대한민국' 행을 골라야 하고,
#   한국 행이 없으면(외국만 있으면) 그 표는 국내 claim에 부적합.
# 도메인 무관 — KOSIS 표준 국가 라벨만 사용, 지표명 하드코딩 없음.

# KOSIS에서 '대한민국'을 가리키는 표기들
_KOREA_LABELS = {"대한민국", "한국", "korea", "republicofkorea", "southkorea", "kor"}

# 국가 차원임을 강하게 시사하는, 한국이 아닌 대표 국가/지역 라벨.
# (이 라벨이 행 카테고리에 보이면 그 컬럼은 '국가 차원'이다)
_FOREIGN_COUNTRY_LABELS = {
    "세계", "아시아", "유럽", "아프리카", "북아메리카", "남아메리카",
    "오세아니아", "일본", "중국", "미국", "독일", "프랑스", "영국",
    "인도", "베트남", "태국", "러시아", "이탈리아", "스페인", "캐나다",
    "호주", "브라질", "멕시코", "인도네시아", "필리핀", "대만", "홍콩",
}


def _is_korea_label(value: str) -> bool:
    """카테고리 값이 '대한민국'을 가리키는지."""
    n = _normalize_korean(value)
    return n in _KOREA_LABELS


def _detect_country_field(rows: list[dict]) -> str | None:
    """행 목록에서 '국가 차원'을 담은 카테고리 컬럼명을 찾는다.

    C1_NM~C4_NM 중, 값들에 외국 국가/대륙 라벨이 하나라도 보이면
    그 컬럼이 국가 차원 — 컬럼명을 반환. 없으면 None.

    표 이름이 아니라 *실제 가져온 데이터*를 보고 판단하므로,
    이름에 국가 표시가 없는 표(DT_2KAA207 등)도 잡아낸다.
    """
    for field_name in ("C1_NM", "C2_NM", "C3_NM", "C4_NM"):
        seen_foreign = False
        seen_any = False
        for r in rows:
            raw = r.get(field_name)
            if raw is None:
                continue
            seen_any = True
            n = _normalize_korean(str(raw))
            if n in {_normalize_korean(x) for x in _FOREIGN_COUNTRY_LABELS}:
                seen_foreign = True
                break
        if seen_any and seen_foreign:
            return field_name
    return None


# ── [수정 v6.24] 데이터 기반 관련성 판정 ──────────────────────────────
# [추가 이유] 가계동향조사 표(DT_1L9V153 '가구원수별 가구당 월평균 가계수지'
#   등)는 표 이름에 '평균소비성향'·'지출 비중' 같은 지표명이 없다. 지표는
#   표 안의 *항목 행*(ITM_NM 등)으로 들어있다. 표 이름만 보는 관련성
#   가드(_is_table_relevant)는 이런 표를 전부 '관련 없음'으로 거부 →
#   가계동향조사 기사 claim이 행 선택까지 가지도 못하고 전멸한다.
#   (출생아 수 표는 이름에 '출생'이 있어 우연히 통과했을 뿐.)
# [해결] 이미 fetch해서 손에 든 rows 안에서 indicator 항목을 직접 찾는다.
#   행에 있으면 그 표는 관련 있는 표 — 표 이름과 무관하게 통과.
# 도메인 무관 — 지표명/표 ID 하드코딩 없음. 행 데이터로만 판별.

def _indicator_in_rows(rows: list[dict], indicator: str | None) -> bool:
    """fetch한 rows의 항목 컬럼에 indicator가 실제로 들어있는지.

    ITM_NM·C1_NM~C4_NM 값을 정규화해서 indicator와 비교한다.
    - 정규화 후 한쪽이 다른 쪽에 포함되면 매칭 (표기차 흡수).
    True면 '표 이름이 안 맞아도 데이터 안에 지표가 있는 표'.

    [패치 M] length ratio guard 추가. 이전엔 row 컬럼이 indicator의 *극히
    일부분*만 포함해도 통과시켜 도메인이 완전히 다른 표를 잘못 인정했다.
    예: indicator='전국 표준단독주택 공시가격 상승률'(정규화 16자) 대해
    C1_NM='전국'(2자, 지역명)만 매칭돼도 → True → 범죄 통계표 통과 →
    부동산 claim에 범죄 row가 evidence로 박히는 가짜 매칭 발생.
    이제 양쪽 길이 비율이 너무 차이나면(짧은 쪽 / 긴 쪽 < 0.5) 거부.
    """
    if not rows or not indicator:
        return False
    ind_norm = _normalize_korean(str(indicator).strip())
    if not ind_norm or len(ind_norm) < 2:
        return False
    _FIELDS = ("ITM_NM", "C1_NM", "C2_NM", "C3_NM", "C4_NM")
    # 매칭 비율 가드: row 컬럼과 indicator의 길이 차이가 2배 이상이면 부정합
    # (지역명/단위 같은 짧은 단어가 긴 indicator 안에 우연히 들어있어
    #  잘못 매칭되는 케이스 차단)
    _MIN_LEN_RATIO = 0.5
    for r in rows:
        if not isinstance(r, dict):
            continue
        for f in _FIELDS:
            raw = r.get(f)
            if raw is None:
                continue
            fn = _normalize_korean(str(raw))
            if not fn:
                continue
            shorter, longer = min(len(fn), len(ind_norm)), max(len(fn), len(ind_norm))
            if shorter / longer < _MIN_LEN_RATIO:
                continue  # 길이 차 너무 큼 — 짧은 키워드 우연 매칭 차단
            # 정규화 후 양방향 substring (예: '평균소비성향' in '가구당평균소비성향')
            if ind_norm in fn or fn in ind_norm:
                return True
    return False


def _select_best_row(
    rows: list[dict],
    indicator: str | None,
    time_period: str | None,
    population: str | None = None,
    unit_hint: str | None = None,
) -> dict | None:
    """KOSIS row 목록에서 indicator + time_period 매칭 row 1개 선택.

    KOSIS row 표준 필드:
        DT: 데이터 값 (string, "20717" 같은)
        PRD_DE: 시점 ("202504", "2025" 등)
        ITM_NM: 항목명 ("출생아 수", "사망자 수", ...)
        C1_NM / C2_NM / C3_NM: 카테고리 차원 (지역, 성별 등)
        UNIT_NM: 단위 ("명", "건", ...)

    매칭 우선순위:
      1. indicator + PRD_DE 둘 다 정확 매칭 (가장 신뢰)
      2. indicator만 매칭, 그중 PRD_DE가 정답 시점에 *포함*되거나 *시작*하는 것
      3. indicator만 매칭 (어떤 시점이든)
      4. PRD_DE만 매칭

    한국어 정규화: '출생아 수' = '출생아수' = '출생아·수'. 공백/특수문자 차이를
    흡수해서 KOSIS 표기차로 매칭 실패하지 않도록 함.

    Returns:
        매칭된 row dict 또는 None.
    """
    if not rows:
        return None

    prd_target = _normalize_prd_de(time_period or "")
    ind_norm_raw = (indicator or "").strip()
    ind_norm = _normalize_korean(ind_norm_raw)
    pop_norm = _normalize_korean((population or "").strip())

    # KOSIS row에서 indicator를 담을 가능성이 있는 모든 string column
    _INDICATOR_FIELDS = ("ITM_NM", "C1_NM", "C2_NM", "C3_NM", "C4_NM")

    def _ind_match(row: dict) -> bool:
        """엄격한 indicator 매칭.

        - 정규화 후 정확 일치 (양방향 ==)
        - 또는 row가 *단일 indicator*일 때만 substring 허용
          ("출생사망혼인이혼" 같은 통합 라벨은 제외)
        """
        if not ind_norm:
            return True
        # KOSIS 통합 카테고리 라벨 (이게 매칭되면 잘못된 row)
        _COMBO_LABELS = (
            "출생사망혼인이혼", "출생사망", "혼인이혼",
            "출생사망혼인", "사망혼인이혼",
        )
        for field_name in _INDICATOR_FIELDS:
            raw = row.get(field_name)
            if raw is None:
                continue
            field_n = _normalize_korean(str(raw))
            if not field_n:
                continue
            # 통합 라벨은 거부
            if field_n in _COMBO_LABELS:
                continue
            # 정확 일치
            if field_n == ind_norm:
                return True
            # 길이 차이가 너무 크면 거부 (예: "출생아수" vs "출생률" 둘 다 짧을 때만 허용)
            len_ratio = max(len(field_n), len(ind_norm)) / max(1, min(len(field_n), len(ind_norm)))
            if len_ratio > 2.5:
                continue
            # 한쪽이 다른 쪽에 완전히 포함되면 OK (예: "출생아수" in "월별출생아수")
            if ind_norm in field_n or field_n in ind_norm:
                return True
        return False

    def _time_match(row: dict) -> bool:
        if not prd_target:
            return True
        prd = str(row.get("PRD_DE", "") or "").strip()
        return prd == prd_target

    # [v6.19] claim 시점 단위 (월/연/분기)
    _claim_gran = _period_granularity(prd_target)

    def _granularity_ok(row: dict) -> bool:
        """[v6.19] claim과 row의 시점 단위가 호환되는지.

        claim이 월(YYYY-MM)인데 row가 연(YYYY) 단위면 → 부적합.
        그 표엔 월 데이터가 없는 것이므로, 연값을 월 claim에
        붙이는 가짜 매칭을 막는다. (예: "9월 24.7도" vs 연평균 14.5도)

        claim이 연 단위거나 claim 시점이 없으면 가드하지 않음.
        """
        if _claim_gran not in ("month", "quarter"):
            return True  # 연 단위 claim — 가드 불필요
        prd = str(row.get("PRD_DE", "") or "").strip()
        row_gran = _period_granularity(prd)
        # PRD_SE='A'(Annual)도 연 단위 신호 — 함께 본다
        prd_se = str(row.get("PRD_SE", "") or "").strip().upper()
        if prd_se == "A":
            row_gran = "year"
        if row_gran == "unknown":
            return True  # 판별 불가 — 막지 않음
        # 월 claim에 연/분기 row, 분기 claim에 연 row → 부적합
        if _claim_gran == "month" and row_gran != "month":
            return False
        if _claim_gran == "quarter" and row_gran == "year":
            return False
        return True

    def _pop_match(row: dict) -> bool:
        if not pop_norm or pop_norm in ("전체", "전국", "계", "total"):
            return True
        for field_name in ("C1_NM", "C2_NM"):
            raw = row.get(field_name)
            if raw is None:
                continue
            field_n = _normalize_korean(str(raw))
            if not field_n:
                continue
            if pop_norm in field_n or field_n in pop_norm:
                return True
        return False

    # ── [수정 v6.25] 단위 적합성 가드 ────────────────────────────────
    # [추가 이유] 가계동향조사 표는 비목 '금액(원)' 행과, 기사 claim의
    #   '지출 비중(%)'·'평균소비성향(%)'이 한 표에 섞여 있다. 단위를
    #   안 보면 ITM='전체가구' 행의 2.253원을 claim '4.8%'와 비교해
    #   가짜 mismatch를 낸다. claim unit_hint와 행 UNIT_NM의 타입이
    #   다르면(% vs 원/명) 그 행은 답이 될 수 없으므로 후보에서 제외.
    # [효과] '지출 비중' claim은 표에 % 행이 없으면 전 행 탈락 → None
    #   → 가짜 mismatch 대신 정직한 unverifiable.
    from structverify.retrieval.kosis_connector import is_same_unit_type
    _unit_hint = (unit_hint or "").strip()

    # ── [패치 3-2b] derived indicator(~증가율 등)면 unit 가드 완화 ──
    # claim이 '출생아 수 증가율'(unit=%) 같은 파생 지표일 때, KOSIS 표엔
    # 증가율 row가 없고 base 값(명/건) row만 있음. unit_hint='%'를 그대로
    # 적용하면 모든 row가 탈락 → 매칭 실패 → unverifiable. loop의
    # _try_growth_rate_from_rows가 prev/current 두 시점 base 값으로 직접
    # 비율을 계산하는 경로가 있으므로, derived indicator에서는 단위 다른
    # base row를 받아들이고 비율 계산은 그 다음 단계에 맡긴다.
    _DERIVED_SUFFIXES = (
        "증가율", "감소율", "증감률", "변화율", "상승률", "하락률",
    )
    _indicator_raw = (indicator or "").strip()
    _is_derived_indicator = any(
        _indicator_raw.endswith(s) for s in _DERIVED_SUFFIXES
    )

    def _unit_match(row: dict) -> bool:
        if not _unit_hint:
            return True  # claim 단위 정보 없음 — 가드 불가
        if _is_derived_indicator:
            return True  # 파생 지표 — base 단위 row를 받아 다음 단계에 맡김
        row_unit = str(row.get("UNIT_NM", "") or "").strip()
        if not row_unit:
            return True  # 행 단위 미상 — 막지 않음
        return is_same_unit_type(_unit_hint, row_unit)

    # ── [수정 v6.23] 국가 차원 필터 ──────────────────────────────────
    # 행 데이터에 국가 차원(세계/일본/중국...)이 있으면, 국내 claim은
    # 반드시 '대한민국' 행이어야 한다. population='전체'라 해도 국가
    # 차원에선 '세계'가 아니라 '대한민국'을 골라야 한다.
    # _pop_match는 '전체'를 무조건 통과시켜 첫 행(세계)을 잡으므로,
    # 그 위에 이 가드를 덧씌운다.
    _country_field = _detect_country_field(rows)
    if _country_field is not None:
        _korea_rows = [
            r for r in rows if _is_korea_label(str(r.get(_country_field, "")))
        ]
        if _korea_rows:
            # 국가 차원이 있는 표 → 한국 행으로만 후보를 좁힌다.
            logger.info(
                f"[_select_best_row] 국가 차원 감지({_country_field}) "
                f"→ 대한민국 행 {len(_korea_rows)}개로 한정"
            )
            rows = _korea_rows
        else:
            # 국가 차원인데 한국 행이 없음 → 해외 전용 표.
            # 국내 claim에는 부적합 → 매칭 실패 처리.
            logger.warning(
                f"[_select_best_row] 국가 차원({_country_field})에 "
                f"대한민국 행 없음 → 해외 전용 표, 국내 claim 부적합"
            )
            return None

    # 1차: indicator + 정확 시점 (+ population + 단위)
    for r in rows:
        if _ind_match(r) and _time_match(r) and _pop_match(r) and _unit_match(r):
            return r

    # 2차: indicator + 정확 시점 + 단위
    if prd_target:
        for r in rows:
            if _ind_match(r) and _time_match(r) and _unit_match(r):
                return r

    # 3차: indicator만 매칭 — 가장 최근 시점 row 우선
    #   [v6.19] 단, claim이 월 단위면 연 단위 row는 제외 (가짜 매칭 방지)
    #   [v6.25] 단위 불일치 row도 제외
    #   [패치 3-2] claim에 명시 시점(prd_target)이 있는데 1·2차에서 정확
    #     매칭 못 했으면, 3차에서 다른 시점 row를 default로 반환하지 않는다.
    #     이전엔 claim '2024-04'인데 표에 4월 row 없으면 1월 row를 reverse
    #     sort로 잡아 evidence 반환 → "(connector default value=21412 → override)"
    #     로그와 함께 calculate가 잘못된 prev값으로 가짜 mismatch를 냈음.
    #     prd_target가 명시된 경우엔 시점 누수보다 None(검증 불가)이 안전 —
    #     호출자가 i'' 자동 fallback으로 다음 candidate를 시도하게 한다.
    if ind_norm and not prd_target:
        matched = [
            r for r in rows
            if _ind_match(r) and _granularity_ok(r) and _unit_match(r)
        ]
        if matched:
            matched.sort(
                key=lambda r: str(r.get("PRD_DE", "") or ""),
                reverse=True,
            )
            return matched[0]

    # 4차: 시점만 매칭 (+ 단위)
    if prd_target:
        for r in rows:
            if _time_match(r) and _unit_match(r):
                return r

    # [v6.19] 월 claim인데 연 단위 표만 있는 경우 — 여기까지 왔으면
    # 단위 호환 row가 하나도 없는 것. None 반환해 검증 불가 처리.
    # (이전엔 3차가 연값을 잡아 "9월 24.7도 vs 연평균 14.5도" 가짜 mismatch 발생)
    return None


def _parse_value(dt_raw: Any) -> float | None:
    """KOSIS DT 필드 → float. 콤마 제거 + 공백 처리."""
    if dt_raw is None:
        return None
    s = str(dt_raw).strip().replace(",", "").replace(" ", "")
    if not s or s in ("-", "X", "x", "..", "..."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ── DataSource ───────────────────────────────────────────────────

@register_datasource("kosis")
class KOSISDataSource(BaseDataSource):
    """KOSIS DataSource — *기존 KOSISConnector* 인스턴스 wrap.

    사용:
        from structverify.retrieval.kosis_connector import KOSISConnector
        connector = KOSISConnector(config=...)
        ds = KOSISDataSource(connector=connector)
        cands = await ds.search_catalog(query="출생아 수")
    """

    name = "kosis"

    # ── [패치 F] _record_cache를 클래스 레벨로 — claim 간 StatRecord 공유 ──
    # _verify_with_agent는 claim마다 KOSISDataSource를 새로 만든다.
    # 인스턴스 별 _record_cache면, claim A의 catalog_search가 캐싱한
    # DT_1B8000G StatRecord(org_id 포함)가 claim B 인스턴스에는 없어,
    # claim B가 prior_success로 DT_1B8000G를 fetch할 때 stat_record=None →
    # connector가 빈 StatRecord(org_id 없음) 생성 → _fetch_with_retry가
    # org_id 없음으로 즉시 None 반환 → 빈 rows → value=None.
    # 클래스 레벨로 공유하면, claim A가 채운 캐시를 claim B가 그대로 본다.
    # (같은 프로세스 안에서만 공유; cross-process는 workspace 영속 필요.)
    _record_cache: dict[str, Any] = {}   # stat_id → StatRecord (class-level)

    def __init__(self, connector: Any = None, **kwargs: Any):
        """
        Args:
            connector: 기존 KOSISConnector 인스턴스. None이면 lazy 생성.
            **kwargs: KOSISConnector 생성 시 사용할 config (connector=None일 때만)
        """
        self._connector = connector
        self._connector_config = kwargs

        logger.info(
            f"[KOSISDataSource] 초기화. connector={'주입됨' if connector else 'lazy 생성 예정'}, "
            f"shared _record_cache 크기={len(KOSISDataSource._record_cache)}"
        )

    def _get_connector(self):
        """기존 connector 인스턴스 반환 — 없으면 lazy 생성."""
        if self._connector is None:
            from structverify.retrieval.kosis_connector import KOSISConnector
            self._connector = KOSISConnector(config=self._connector_config)
            logger.info("[KOSISDataSource] KOSISConnector lazy 생성됨")
        return self._connector

    def _make_query(self, query_str: str, **kwargs) -> Any:
        """ConnectorQuery 생성 — 사용자 코드의 dataclass."""
        from structverify.retrieval.base_connector import ConnectorQuery
        return ConnectorQuery(
            keyword=query_str,
            indicator=kwargs.get("indicator", query_str),
            time_period=kwargs.get("time_period"),
            population=kwargs.get("population"),
            extra_params=kwargs.get("extra_params", {}),
        )

    async def _lookup_stat_record_by_id(self, stat_id: str) -> Any | None:
        """[패치 F-2] kosis_stat_catalog 테이블에서 stat_id로 StatRecord 직접 조회.

        용도: prior_success로 들어온 stat_id가 클래스 캐시에도 없을 때
              (cross-process / restart) DB에서 한 번 끌어와 placeholder
              fetch가 빈 rows로 침묵 실패하는 걸 방지.
        """
        if not stat_id:
            return None
        try:
            import asyncpg
        except ImportError:
            return None
        connector = self._get_connector()
        catalog = getattr(connector, "catalog", None)
        pg_dsn = getattr(catalog, "pg_dsn", None) if catalog else None
        if not pg_dsn:
            return None
        try:
            conn = await asyncpg.connect(pg_dsn)
        except Exception as e:
            logger.debug(f"[_lookup_stat_record_by_id] DB 연결 실패: {e}")
            return None
        try:
            row = await conn.fetchrow(
                """
                SELECT stat_id, stat_name, org_id, org_name, category_path, keywords
                FROM kosis_stat_catalog
                WHERE stat_id = $1
                LIMIT 1
                """,
                stat_id,
            )
        except Exception as e:
            logger.debug(f"[_lookup_stat_record_by_id] 쿼리 실패: {e}")
            row = None
        finally:
            try:
                await conn.close()
            except Exception:
                pass
        if not row:
            return None
        from structverify.retrieval.base_connector import StatRecord
        return StatRecord(
            stat_id=row["stat_id"],
            stat_name=row["stat_name"],
            org_id=row["org_id"],
            org_name=row["org_name"] or "",
            available_periods=[],
            relevance_score=1.0,
            metadata={
                "source": "pgvector_lookup_by_id",
                "category_path": row.get("category_path"),
                "keywords": row.get("keywords"),
            },
        )

    # ── BaseDataSource 인터페이스 ──

    async def search_catalog(
        self,
        query: str,
        category: list[str] | None = None,
        top_k: int = 10,
    ) -> list[CatalogCandidate]:
        """KOSIS 카탈로그 검색. KOSISConnector.search(ConnectorQuery) 호출."""
        connector = self._get_connector()
        cq = self._make_query(query, extra_params={"category": category} if category else {})

        try:
            records = await connector.search(cq)
        except Exception as e:
            logger.warning(f"[KOSISDataSource] search 실패: {type(e).__name__}: {e}")
            return []

        # StatRecord → CatalogCandidate
        candidates: list[CatalogCandidate] = []
        for r in (records or [])[:top_k]:
            sid = str(getattr(r, "stat_id", ""))
            if sid:
                self._record_cache[sid] = r   # 캐시 저장
            candidates.append({
                "id": sid,
                "name": str(getattr(r, "stat_name", "")),
                "score": float(getattr(r, "relevance_score", 0.0) or 0.0),
                "raw": r,
            })

        logger.info(
            f"[KOSISDataSource] search_catalog(query={query!r}): {len(candidates)}개 후보"
        )
        return candidates

    async def fetch_evidence(
        self,
        candidate_id: str,
        params: dict[str, Any] | None = None,
    ) -> EvidenceData | None:
        """KOSIS에서 실제 수치 조회. KOSISConnector.fetch(stat_id, params) 호출.

        Phase D: rows 안에서 indicator + time_period 매칭 row를 직접 골라
                 정확한 value/unit/time을 EvidenceData에 담아 반환.
        """
        connector = self._get_connector()
        params = params or {}

        # stat_record가 params에 없으면 캐시에서 복원
        if not params.get("stat_record"):
            cached = self._record_cache.get(candidate_id)
            if cached is not None:
                params = {**params, "stat_record": cached}
                logger.info(
                    f"[KOSISDataSource] stat_record 캐시 적중: {candidate_id} "
                    f"(org_id={getattr(cached, 'org_id', None)!r})"
                )
            else:
                # ── [패치 F-2] 캐시 미스 → DB에서 stat_id 직접 lookup ──
                # 클래스 레벨 캐시(F)에도 없으면(예: 다른 프로세스/restart),
                # pgvector 카탈로그 DB에서 stat_id로 직접 StatRecord를 조회.
                # 없으면 placeholder StatRecord로 fetch가 빈 rows 반환 →
                # value=None 침묵 실패가 나므로, 여기서 한 번 강제 enrich한다.
                try:
                    fetched = await self._lookup_stat_record_by_id(candidate_id)
                except Exception as e:
                    logger.debug(
                        f"[KOSISDataSource] stat_id lookup 실패: {candidate_id} — {e}"
                    )
                    fetched = None
                if fetched is not None:
                    self._record_cache[candidate_id] = fetched   # 다음 호출 캐시
                    params = {**params, "stat_record": fetched}
                    logger.info(
                        f"[KOSISDataSource] stat_record DB 복원 성공: {candidate_id} "
                        f"(org_id={getattr(fetched, 'org_id', None)!r})"
                    )
                else:
                    logger.warning(
                        f"[KOSISDataSource] stat_record 캐시·DB 모두 미스: {candidate_id} "
                        f"— connector가 placeholder로 fetch 시도하지만 org_id 없어 실패 가능"
                    )

        cq = params.get("query")
        if cq is None:
            cq = self._make_query(
                params.get("query_str", candidate_id),
                indicator=params.get("indicator"),
                time_period=params.get("time_period"),
                population=params.get("population"),
            )

        # fetch params 구성
        fetch_params = {
            "time_period": params.get("time_period"),
            "population": params.get("population"),
            "query": cq,
            "stat_record": params.get("stat_record"),
            "prdSe": params.get("prdSe", "Y"),
            "startPrdDe": params.get("startPrdDe", ""),
            "endPrdDe": params.get("endPrdDe", ""),
        }
        fetch_params.update({k: v for k, v in params.items() if k not in fetch_params})

        # time_period → KOSIS API 파라미터 자동 변환
        # [패치] 이전엔 `not params.get("prdSe")`로 가드해서, prdSe가 명시되면
        # _parse_time_period가 발동 안 하고 params에 들어온 잘못된 startPrdDe
        # ('202401' 같은 1월 값)가 그대로 KOSIS에 전송 → 4월 row 못 받음.
        # 이제 time_period가 있으면 무조건 _parse_time_period 결과로 덮어씀.
        # 변환에 성공한 경우에만 적용 (잘못된 형식은 기존 값 유지).
        tp = params.get("time_period")
        if tp:
            prd_se, start, end = _parse_time_period(tp)
            if prd_se and start and end:
                fetch_params["prdSe"] = prd_se
                fetch_params["startPrdDe"] = start
                fetch_params["endPrdDe"] = end
                logger.info(
                    f"[KOSISDataSource] time_period={tp!r} → "
                    f"prdSe={prd_se!r} startPrdDe={start!r} endPrdDe={end!r} "
                    f"(params에 prdSe={params.get('prdSe')!r} "
                    f"startPrdDe={params.get('startPrdDe')!r} 있어도 덮어씀)"
                )
            # ── [v6.17] growth_rate 범위 확장 — prev 시점까지 받아오기 ──
            # fetch_evidence 도구가 _range_start/_range_end를 넣어줬으면
            # startPrdDe/endPrdDe를 그 범위로 덮어쓴다. 현재+이전 시점을
            # 한 번에 받아 loop이 증가율을 직접 계산할 수 있게 함.
            rng_start = params.get("_range_start")
            rng_end = params.get("_range_end")
            if rng_start and rng_end:
                _ps, _s, _ = _parse_time_period(str(rng_start))
                _, _e, _ = _parse_time_period(str(rng_end))
                if _s and _e:
                    fetch_params["prdSe"] = _ps
                    fetch_params["startPrdDe"] = _s
                    fetch_params["endPrdDe"] = _e
                    logger.info(
                        f"[KOSISDataSource] growth_rate 범위 적용: "
                        f"prdSe={_ps!r} startPrdDe={_s!r} endPrdDe={_e!r} "
                        f"(time_period={tp!r}, prev까지 확장)"
                    )
                else:
                    logger.info(
                        f"[KOSISDataSource] time_period {tp!r} → "
                        f"prdSe={prd_se!r} startPrdDe={start!r} endPrdDe={end!r}"
                    )
            else:
                logger.info(
                    f"[KOSISDataSource] time_period {tp!r} → "
                    f"prdSe={prd_se!r} startPrdDe={start!r} endPrdDe={end!r}"
                )

        try:
            data = await connector.fetch(candidate_id, fetch_params)
        except Exception as e:
            logger.warning(
                f"[KOSISDataSource] fetch({candidate_id}) 실패: {type(e).__name__}: {e}"
            )
            return None

        if data is None:
            return None

        # raw_response에서 rows 추출 (KOSIS API 응답 형식)
        raw = getattr(data, "raw_response", None) or {}
        if isinstance(raw, dict):
            rows = raw.get("row") or raw.get("rows") or []
        else:
            rows = []

        stat_id_str = str(getattr(data, "stat_id", candidate_id))
        stat_name_str = str(getattr(data, "stat_name", "")) or candidate_id

        # ── [v6.17] 테이블 관련성 체크 ──────────────────────────────────
        # catalog top 후보를 무조건 fetch하면 "합계출산율 - 동북·중앙아시아"
        # 같은 엉뚱한 표를 한국 claim에 가져오게 됨. claim의 indicator/population이
        # 표 이름과 같은 분야인지 검사해서, 무관하면 fetch 거부 (None 반환).
        from structverify.retrieval.kosis_connector import _is_table_relevant
        claim_indicator = (params.get("indicator") or "").strip()
        claim_population = (params.get("population") or "").strip()
        # indicator + population 합쳐서 검사 (지역명이 population에 있을 수 있음)
        relevance_query = f"{claim_indicator} {claim_population}".strip()

        # (a) 해외 지역 표 거르기 — "합계출산율 - 동북·중앙아시아" 처럼
        #     지표명은 같지만 한국 통계가 아닌 표. claim에 해당 지역어가
        #     없는데 표 이름에 있으면 fetch 거부.
        _OVERSEAS = (
            "동북·중앙아시아", "남부·동남아시아", "중앙아시아", "동남아시아",
            "북아메리카", "남아메리카", "유럽", "아프리카", "오세아니아",
            "oecd", "g20", "세계", "국제", "해외",
        )
        _stat_lower = stat_name_str.lower()
        _claim_lower = relevance_query.lower()
        for kw in _OVERSEAS:
            if kw in _stat_lower and kw not in _claim_lower:
                logger.warning(
                    f"[KOSISDataSource] 해외/국제 통계표 → fetch 거부: "
                    f"[{stat_id_str}] {stat_name_str!r} "
                    f"(claim은 국내 통계: {relevance_query!r})"
                )
                return None

        # (a-2) 특수 관측 표 거르기 — "[해양기상] 등표 관측값",
        #     "[항공기상] 공항기상관측", "[고층기상] 레윈존데" 등.
        #     일반 "연평균기온" claim에 해양/항공 관측 표를 가져오면
        #     육상 통계와 안 맞아 가짜 mismatch가 남. claim에 해당
        #     관측 도메인 단어가 없으면 fetch 거부.
        _SPECIALIZED = (
            "해양기상", "항공기상", "고층기상", "등표", "레윈존데",
            "공항기상", "부이", "방재기상",
        )
        for kw in _SPECIALIZED:
            if kw in _stat_lower and kw not in _claim_lower:
                logger.warning(
                    f"[KOSISDataSource] 특수 관측 통계표 → fetch 거부: "
                    f"[{stat_id_str}] {stat_name_str!r} "
                    f"(claim은 일반 통계: {relevance_query!r})"
                )
                return None

        # (b) 일반 관련성 체크 — indicator가 표 이름과 같은 분야인지
        # [수정 v6.24] 표 이름만 보던 것을 데이터 기반으로 보강.
        # 가계동향조사 표처럼 지표가 표 이름이 아니라 *행 항목*(ITM_NM 등)
        # 으로 들어있는 표는 _is_table_relevant(이름 기반)가 거부한다.
        # → 먼저 이미 fetch한 rows 안에 indicator가 있는지 확인하고,
        #   있으면 '데이터에 지표가 실재하는 관련 표'이므로 통과시킨다.
        #   rows에 없을 때만 기존 이름 기반 가드로 폴백.
        if relevance_query and not _is_table_relevant(relevance_query, stat_name_str):
            if _indicator_in_rows(rows, claim_indicator):
                logger.info(
                    f"[KOSISDataSource] 표 이름엔 지표 없으나 행 데이터에 "
                    f"'{claim_indicator}' 존재 → 관련 표로 인정: [{stat_id_str}]"
                )
            elif params.get("_from_prior_success"):
                # [패치 E] 같은 job의 다른 claim이 이미 fetch 성공한 표는 표
                # 이름이 indicator와 안 닿더라도 row 안에 indicator가 있을
                # 가능성이 입증됨. row 매칭은 _select_best_row가 책임진다.
                # (안전성: row 매칭이 None을 반환하면 어차피 evidence는 None.)
                logger.info(
                    f"[KOSISDataSource] 표 관련성 가드 우회 (prior_success): "
                    f"[{stat_id_str}] — row 매칭 단계에서 indicator 검증"
                )
            else:
                logger.warning(
                    f"[KOSISDataSource] 테이블 관련성 없음 → fetch 거부: "
                    f"[{stat_id_str}] {stat_name_str!r} vs "
                    f"indicator={claim_indicator!r} population={claim_population!r}"
                )
                return None

        # ★ rows에서 indicator + time_period 매칭 row 직접 선택
        # connector가 drows[0]만 official_value로 만들기 때문에 통합표에서 잘못된 row를 받음.
        # 여기서 rows 전체를 보고 정확한 row를 찾아 value/unit/time override.
        connector_value = getattr(data, "official_value", None)
        connector_unit = getattr(data, "unit", "") or ""
        connector_time = getattr(data, "time_period", "") or ""

        # [v6.20] 표 단위 적합성 사전 체크 — claim이 월(YYYY-MM)인데
        # 표가 연 단위 데이터만 있으면, 그 표엔 월 데이터가 없는 것.
        # fetch해도 가짜 매칭(연값을 월 claim에)만 나오므로 일찍 거부.
        # 도메인 무관: KOSIS 표별 수록주기 차이를 행 데이터로 판별.
        _claim_period = params.get("time_period") or ""
        if rows and not _table_has_period_for(rows, _claim_period):
            logger.warning(
                f"[KOSISDataSource] 표 시점 단위 불일치 → fetch 거부: "
                f"[{stat_id_str}] {stat_name_str!r} — "
                f"claim 시점은 '{_claim_period}'(월/분기)인데 "
                f"표에 해당 단위 데이터 없음 (연 단위 표). evidence 없음 처리."
            )
            return None

        best_row = None
        if rows:
            best_row = _select_best_row(
                rows,
                indicator=params.get("indicator"),
                time_period=params.get("time_period"),
                population=params.get("population"),
                unit_hint=params.get("unit_hint"),
            )

        if best_row is not None:
            matched_value = _parse_value(best_row.get("DT"))
            matched_unit = str(best_row.get("UNIT_NM", "") or "").strip()
            matched_time = str(best_row.get("PRD_DE", "") or "").strip()
            matched_indicator = str(best_row.get("ITM_NM", "") or "").strip()

            if matched_value is not None:
                logger.info(
                    f"[KOSISDataSource] row 매칭 성공: "
                    f"ITM_NM={matched_indicator!r} PRD_DE={matched_time!r} "
                    f"DT={matched_value} UNIT={matched_unit!r} "
                    f"(connector default value={connector_value} → override, {len(rows)} rows 중)"
                )
                return {
                    "value": matched_value,
                    "unit": matched_unit,
                    "time_period": matched_time,
                    "source": "kosis",
                    "stat_table_id": stat_id_str,
                    "stat_name": stat_name_str,
                    "rows": rows,
                    "matched_row": best_row,
                    "matched_indicator": matched_indicator,
                    "raw": data,
                }
            else:
                logger.warning(
                    f"[KOSISDataSource] row 매칭은 됐지만 DT 파싱 실패: "
                    f"DT={best_row.get('DT')!r}"
                )
        else:
            # ★ 매칭 실패 시 row sample 로그 — KOSIS 표 column 구조 파악용 (한 번만 보면 됨)
            if rows:
                all_keys = list((rows[0] or {}).keys())
                sample_keys = all_keys[:10]
                logger.warning(
                    f"[KOSISDataSource] indicator/time 매칭 row 못 찾음 "
                    f"(rows={len(rows)}, indicator={params.get('indicator')!r}, "
                    f"time={params.get('time_period')!r}) — evidence 없음 처리."
                )
                logger.warning(f"  row[0] 전체 키({len(all_keys)}개): {all_keys}")
                sample_prds = sorted(
                    set(str(r.get("PRD_DE", "MISSING")) for r in rows[:200])
                )
                logger.warning(
                    f"  PRD_DE 분포 (상위 200 row 중 unique {len(sample_prds)}개, "
                    f"앞 20개): {sample_prds[:20]}"
                )
                for i, r in enumerate(rows[:3]):
                    snippet = {k: r.get(k) for k in sample_keys if k in r}
                    logger.warning(f"  row[{i}]: {snippet}")
                # [v6.19] rows는 있는데 claim에 맞는 row가 없음 →
                # connector default 값(rows[0] 등 엉뚱한 값)을 쓰면
                # "9월 24.7도 vs 연평균 14.5도" 같은 가짜 mismatch가 남.
                # default로 둔갑시키지 말고 evidence 없음(None)으로 반환.
                return None
            else:
                logger.info(
                    f"[KOSISDataSource] rows 비어있음 (indicator={params.get('indicator')!r}, "
                    f"time={params.get('time_period')!r}) — connector default 값 사용"
                )

        # fallback: rows가 아예 없을 때만 connector가 골라준 값 사용
        return {
            "value": connector_value,
            "unit": connector_unit,
            "time_period": connector_time,
            "source": "kosis",
            "stat_table_id": stat_id_str,
            "stat_name": stat_name_str,
            "rows": rows,
            "raw": data,
        }

    def supports_time_filter(self) -> bool:
        return True

    def supports_population_filter(self) -> bool:
        return False  # KOSIS는 표마다 population 분리 (별도 dim)

    async def close(self) -> None:
        pass