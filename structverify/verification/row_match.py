"""[리팩] KOSIS row·시점 매칭 — verifier._find_best_match 등 분리 (fallback 프로필)"""
from __future__ import annotations

from structverify.utils.logger import get_logger

from .units import is_same_unit_type, normalize_value

logger = get_logger(__name__)


def extract_numeric_values(rows: list[dict]) -> list[dict]:
    """raw_response["row"]에서 수치/단위/기간 추출"""
    values = []
    for row in rows:
        dt = row.get("DT", "")
        unit = row.get("UNIT_NM", "")
        prd = row.get("PRD_DE", "")
        try:
            val = float(str(dt).replace(",", ""))
            values.append({"value": val, "unit": unit, "period": prd, "raw": row})
        except (ValueError, TypeError):
            continue
    return values


def normalize_period(period: str) -> str:
    """KOSIS PRD_DE 다양한 형식을 *YYYYMM* 또는 *YYYY*로 정규화.

    지원 형식:
      "202504"   (6자 숫자)           → "202504"
      "2025-04"  (7자 하이픈)         → "202504"
      "2025.04"  (7자 점)             → "202504"
      "2025M04"  (7자 M 구분)         → "202504"
      "2025/04"                       → "202504"
      "2025"     (4자 — 연간 누계)    → "2025"
      "2025Q1"   (분기 — Q 포함)      → "2025"
      "202504XX" (8자+ — 일별 등)     → "202504"
    """
    if not period:
        return ""
    p = str(period).strip()

    # 분기 처리: "2025Q1", "20251Q" 등 → "2025"
    if "Q" in p.upper():
        return p[:4] if p[:4].isdigit() else ""

    clean = "".join(c for c in p if c.isdigit())

    if len(clean) >= 6:
        return clean[:6]
    if len(clean) == 4:
        return clean
    return clean


def period_is_monthly(period: str) -> bool:
    """이 period가 *월 단위* row인지 (claim_year_month와 정확 비교 가능한 형식)."""
    normalized = normalize_period(period)
    return len(normalized) == 6 and normalized.isdigit()


def period_is_annual(period: str) -> bool:
    """이 period가 *연간 누계 또는 연도 단위* row인지.

    claim이 *월값*인데 (claim_year_month 있음) 매칭되면 *후순위*로 처리해야 함.
    """
    normalized = normalize_period(period)
    return len(normalized) == 4 and normalized.isdigit()


def period_matches_ym(period: str, claim_year_month: str) -> bool:
    """정규화 후 claim_year_month와 정확히 매칭되는지 (tier 1 후보)."""
    if not period or not claim_year_month:
        return False
    normalized = normalize_period(period)
    # claim_ym도 정규화 (혹시 "2025-04" 형식으로 들어올 수 있음)
    claim_norm = normalize_period(claim_year_month)
    if len(claim_norm) != 6 or len(normalized) < 6:
        return False
    return normalized[:6] == claim_norm[:6]


def find_best_match(
    claimed: float,
    claim_unit: str,
    claim_year: str | None,
    kosis_values: list[dict],
    claim_year_month: str | None = None,
) -> tuple[dict | None, float]:
    """
    KOSIS 전체 행에서 claim과 가장 가까운 값 탐색.
    factcheck_test.py v7 numeric_check 로직 그대로 (fallback 프로필).

    [v6.14 G fix] all_rows_empty — 지표명-단위 일체형 표면 row.unit 비어도 통과.
    [v6.14 F1 fix] claim_year_month 있으면 동일 연-월 row를 *최우선* picking.
    [v6.15] tier 2a (월 row) / 2b (연간 누계) 분리.
    """
    # [v6.14 G fix] 표 전체 row.unit 분포 분석
    nonempty_units = [
        kv["unit"] for kv in kosis_values
        if kv.get("unit") and str(kv["unit"]).strip()
    ]
    all_rows_empty = len(nonempty_units) == 0

    total_rows = len(kosis_values)
    year_filtered = 0
    unit_filtered = 0
    zero_filtered = 0

    tier1_candidates: list[dict] = []
    tier2a_candidates: list[dict] = []
    tier2b_candidates: list[dict] = []
    tier3_candidates: list[dict] = []

    for kv in kosis_values:
        kv_year = None
        kv_period = kv.get("period") or ""
        if claim_year and kv_period:
            kv_year = kv_period[:4]
            try:
                # 연도 정확 일치 필터 (박재윤 2026-05-14: ±2 → 0으로 변경)
                if abs(int(claim_year) - int(kv_year)) > 0:
                    year_filtered += 1
                    continue
            except (ValueError, TypeError):
                pass

        normalized = normalize_value(kv["value"], kv["unit"])
        if normalized == 0:
            zero_filtered += 1
            continue

        if not is_same_unit_type(claim_unit, kv["unit"], all_rows_empty=all_rows_empty):
            unit_filtered += 1
            continue

        # [v6.14 H fix] 상대 오차 — 분모 1 버그 회피 (소수 지표 오판정 방지)
        denom = max(abs(normalized), abs(claimed), 1e-9)
        error_rate = abs(normalized - claimed) / denom
        kv_with_meta = {**kv, "normalized": normalized, "error_rate": error_rate}

        # [F1] 시점 tier 분류 — [v6.15] period 정규화 + 연간/월 row 분리
        if claim_year_month and kv_period:
            if period_matches_ym(kv_period, claim_year_month):
                tier1_candidates.append(kv_with_meta)
            elif claim_year and kv_period.startswith(claim_year):
                if period_is_annual(kv_period):
                    # 연간 누계 — claim이 월값일 때 후순위
                    tier2b_candidates.append(kv_with_meta)
                else:
                    tier2a_candidates.append(kv_with_meta)
            else:
                tier3_candidates.append(kv_with_meta)
        elif claim_year and kv_period and kv_period.startswith(claim_year):
            tier2a_candidates.append(kv_with_meta)
        else:
            tier3_candidates.append(kv_with_meta)

    selected_tier = None
    pool: list[dict] = []
    if tier1_candidates:
        pool = tier1_candidates
        selected_tier = "1 (동일 연-월)"
    elif tier2a_candidates:
        pool = tier2a_candidates
        selected_tier = "2a (동일 연도, 월 row)"
    elif tier2b_candidates:
        pool = tier2b_candidates
        selected_tier = "2b (동일 연도, 연간 누계 — claim 월값과 시점 mismatch 가능)"
    elif tier3_candidates:
        pool = tier3_candidates
        selected_tier = "3 (±2년)"

    # [v6.15] 선택된 tier 번호를 각 후보에 기록 (verdict 가드용)
    _tier_num = 1
    if selected_tier:
        if selected_tier.startswith("2"):
            _tier_num = 2
        elif selected_tier.startswith("3"):
            _tier_num = 3
    for kv in pool:
        kv["_tier"] = _tier_num

    best_match = None
    best_error = float("inf")
    for kv in pool:
        if kv["error_rate"] < best_error:
            best_error = kv["error_rate"]
            best_match = kv

    candidates = len(pool)

    logger.info(
        f"[verifier] match 탐색: claim={claimed}/{claim_unit!r} year={claim_year} "
        f"ym={claim_year_month} | 전체 row={total_rows} (all_rows_empty={all_rows_empty}) → "
        f"연도제외={year_filtered}, zero제외={zero_filtered}, 단위불일치제외={unit_filtered} | "
        f"tier1(연-월)={len(tier1_candidates)}, tier2a(연도+월)={len(tier2a_candidates)}, "
        f"tier2b(연간누계)={len(tier2b_candidates)}, tier3(±2년)={len(tier3_candidates)} → "
        f"선택 tier={selected_tier}, 최종 후보={candidates}"
    )
    if best_match:
        logger.info(
            f"[verifier] best_match: period={best_match.get('period')!r} "
            f"unit={best_match.get('unit')!r} value={best_match.get('value')} "
            f"(normalized={best_match.get('normalized')}) "
            f"error={best_match.get('error_rate'):.4f}"
        )
    else:
        logger.info("[verifier] best_match: 없음 (단위/연도 필터 통과 row 없음)")

    return best_match, best_error


# ── [패치 H-3] agent 프로필 row pool (loop.py에서 추출) ─────────────────────
# matched_row의 ITM_NM·C1_NM~C4_NM을 criteria로 추출해, aggregated rows
# 풀에서 같은 지표에 다른 시점(target_time)의 row를 찾는다.
# 시점만 보고 row를 잡으면 다른 지표 row(출생아 수 vs 혼인 건수 등)가
# 잘못 매칭되어 가짜 prev/current 비교를 만든다 — criteria 필터로 차단.

INDICATOR_CRITERIA_FIELDS = ("ITM_NM", "C1_NM", "C2_NM", "C3_NM", "C4_NM")


def parse_row_dt(raw) -> float | None:
    """KOSIS row의 DT 필드를 float로 파싱 (콤마/공백 제거)."""
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def find_row_value_for_time(rows: list, target_time: str) -> float | None:
    """[v6.17] KOSIS 표 rows에서 특정 시점(PRD_DE) 행의 값(DT)을 찾는다.

    growth_rate 직접 계산용 — 같은 표에서 prev 시점 값을 추출한다.
    target_time: 'YYYY' 또는 'YYYY-MM'. PRD_DE는 'YYYY' 또는 'YYYYMM' 형식.
    """
    if not rows or not target_time:
        return None
    # 'YYYY-MM' → 'YYYYMM' 정규화
    norm = str(target_time).replace("-", "").strip()
    for row in rows:
        if not isinstance(row, dict):
            continue
        prd = str(row.get("PRD_DE", "") or "").strip()
        if prd == norm:
            v = parse_row_dt(row.get("DT"))
            if v is not None:
                return v
    # 연도만으로 재시도 (target이 'YYYY-MM'인데 표는 연 단위인 경우)
    year = norm[:4]
    if year and year != norm:
        for row in rows:
            if not isinstance(row, dict):
                continue
            prd = str(row.get("PRD_DE", "") or "").strip()
            if prd == year:
                v = parse_row_dt(row.get("DT"))
                if v is not None:
                    return v
    return None


def extract_criteria_from_row(row: dict) -> dict:
    """matched_row에서 지표 식별 컬럼만 추출."""
    if not isinstance(row, dict):
        return {}
    return {
        k: row[k]
        for k in INDICATOR_CRITERIA_FIELDS
        if k in row and row[k] is not None and str(row[k]).strip() != ""
    }


def find_value_for_time_with_criteria(
    all_rows: list[dict],
    target_time: str,
    criteria: dict | None,
) -> tuple[float, dict] | None:
    """rows[]에서 target_time 매칭 + criteria 컬럼 값 일치하는 row 찾기.

    criteria가 비면 find_row_value_for_time과 동일 동작.
    찾으면 (DT 값, 매칭한 row) 반환.
    """
    if not all_rows or not target_time:
        return None
    norm = str(target_time).replace("-", "").strip()

    def _row_matches_criteria(row: dict) -> bool:
        if not criteria:
            return True
        for k, v in criteria.items():
            if str(row.get(k, "")).strip() != str(v).strip():
                return False
        return True

    # 1차: PRD_DE 완전 일치 + criteria 일치
    for row in all_rows:
        if not isinstance(row, dict):
            continue
        prd = str(row.get("PRD_DE", "") or "").strip()
        if prd != norm:
            continue
        if not _row_matches_criteria(row):
            continue
        v = parse_row_dt(row.get("DT"))
        if v is not None:
            return (v, row)

    # 2차: 연 단위 fallback (PRD_DE='YYYY')
    year = norm[:4]
    if year and year != norm:
        for row in all_rows:
            if not isinstance(row, dict):
                continue
            prd = str(row.get("PRD_DE", "") or "").strip()
            if prd != year:
                continue
            if not _row_matches_criteria(row):
                continue
            v = parse_row_dt(row.get("DT"))
            if v is not None:
                return (v, row)
    return None


def aggregate_rows_from_fetches(fetch_observations: list) -> list[dict]:
    """여러 fetch observation rows[] 평탄화 (loop._aggregate_rows_from_fetches)."""
    out: list[dict] = []
    if not fetch_observations:
        return out
    seen_ids: set[int] = set()
    for obs in fetch_observations:
        ev = (getattr(obs, "output", None) or {}).get("evidence") or {}
        rs = ev.get("rows") or []
        for r in rs:
            if not isinstance(r, dict):
                continue
            rid = id(r)
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            out.append(r)
    return out
