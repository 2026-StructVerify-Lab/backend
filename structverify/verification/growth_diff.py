"""[리팩] 증가율/차이 자동 계산 — verifier._verify_growth_or_diff 분리 (fallback 프로필)"""
from __future__ import annotations

from collections.abc import Callable

from structverify.core.schemas import Claim, Evidence, MismatchType, VerificationResult
from structverify.utils.logger import get_logger

from .row_match import (
    aggregate_rows_from_fetches,
    extract_criteria_from_row,
    extract_numeric_values,
    find_row_value_for_time,
    find_value_for_time_with_criteria,
    parse_row_dt,
    period_is_annual,
    period_matches_ym,
)
from .units import normalize_value
from .verdict_thresholds import verdict_from_error

logger = get_logger(__name__)


def verify_growth_or_diff(
    claim: Claim,
    evidence: Evidence,
    claim_year: str | None,
    claim_year_month: str | None,
    prev_value: float,
    is_ratio: bool,
    config: dict,
    classify_mismatch: Callable[[Claim, Evidence, float, dict], MismatchType],
) -> VerificationResult | None:
    """
    증가율/차이 schema의 자동 계산 검증.

    KOSIS 현재 시점 row를 못 찾으면 None 반환 (호출자가 일반 분기로 fallthrough).
    """
    claimed = claim.schema.value
    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    rows = raw.get("row", [])
    if not isinstance(rows, list) or not rows:
        return None

    kosis_values = extract_numeric_values(rows)
    if not kosis_values:
        return None

    tier1, tier2a, tier2b, tier3 = [], [], [], []
    for kv in kosis_values:
        kv_period = kv.get("period") or ""
        normalized = normalize_value(kv["value"], kv["unit"])
        if normalized == 0:
            continue
        if claim_year and kv_period:
            try:
                if abs(int(claim_year) - int(kv_period[:4])) > 2:
                    continue
            except (ValueError, TypeError):
                pass
        kv_norm = {**kv, "normalized": normalized}
        if claim_year_month and period_matches_ym(kv_period, claim_year_month):
            tier1.append(kv_norm)
        elif claim_year and kv_period.startswith(claim_year):
            if claim_year_month and period_is_annual(kv_period):
                tier2b.append(kv_norm)
            else:
                tier2a.append(kv_norm)
        else:
            tier3.append(kv_norm)

    pool = tier1 or tier2a or tier2b or tier3
    if not pool:
        logger.info("[verifier C2] 증가율 계산: KOSIS에서 현재 시점 row 못 찾음. fallthrough.")
        return None

    def _scale_match(kv):
        v = abs(kv["normalized"])
        p = abs(prev_value)
        if v == 0 or p == 0:
            return float("inf")
        return abs(v / p - 1) if v >= p else abs(p / v - 1)

    current_row = min(pool, key=_scale_match)
    current_value = current_row["normalized"]

    if is_ratio:
        calculated = (current_value - prev_value) / prev_value * 100
        calc_desc = (
            f"증가율 ({current_value} - {prev_value}) / {prev_value} * 100 = {calculated:.2f}%"
        )
    else:
        calculated = current_value - prev_value
        calc_desc = f"차이 {current_value} - {prev_value} = {calculated:.4f}"

    denom = max(abs(calculated), abs(claimed), 1e-9)
    error_rate = abs(calculated - claimed) / denom

    logger.info(
        f"[verifier C2] {calc_desc} | claim={claimed} | error_rate={error_rate*100:.2f}% | "
        f"current_row: period={current_row.get('period')!r} value={current_row['value']} "
        f"unit={current_row.get('unit')!r}"
    )

    evidence = evidence.model_copy(update={
        "official_value": current_row.get("value"),
        "unit": current_row.get("unit") or evidence.unit,
        "time_period": current_row.get("period") or evidence.time_period,
    })

    best_match_info = {
        **current_row,
        "error_rate": error_rate,
        "calculated_from_prev": calculated,
        "prev_value": prev_value,
    }
    return verdict_from_error(
        claim, evidence, error_rate, best_match_info, config, classify_mismatch,
    )


# ── agent 프로필 rows pool 계산 (loop._try_* 에서 추출) ─────────────────────

def try_growth_rate_from_rows(
    evidence: dict,
    schema,
    claim_id: str,
    all_fetch_observations: list | None = None,
) -> tuple[float, float, float, str] | None:
    """growth_rate claim — 표 rows에서 (current-prev)/prev*100 직접 계산."""
    prev_time = getattr(schema, "prev_time_period", None) if schema else None
    if not prev_time:
        return None

    cur_time = getattr(schema, "time_period", None) if schema else None

    rows = list(evidence.get("rows") or [])
    pool_rows: list[dict] = []
    if all_fetch_observations:
        pool_rows = aggregate_rows_from_fetches(all_fetch_observations)
    for r in rows:
        if isinstance(r, dict) and r not in pool_rows:
            pool_rows.append(r)
    if not pool_rows:
        return None

    matched_row = evidence.get("matched_row") or {}
    criteria = extract_criteria_from_row(matched_row)

    current_val: float | None = None
    if cur_time:
        cur_hit = find_value_for_time_with_criteria(pool_rows, cur_time, criteria)
        if cur_hit is not None:
            current_val, _ = cur_hit
    if current_val is None:
        current_val = parse_row_dt(evidence.get("value"))
        if current_val is None and matched_row:
            current_val = parse_row_dt(matched_row.get("DT"))
    if current_val is None:
        return None

    prev_hit = find_value_for_time_with_criteria(pool_rows, prev_time, criteria)
    if prev_hit is None:
        logger.info(
            f"[loop] {claim_id}: growth_rate 직접계산 — criteria 매칭 prev row "
            f"{prev_time!r} 못 찾음. 지표 무관 시점 매칭으로 fallback "
            f"(pool={len(pool_rows)} rows, criteria={list(criteria.keys()) or '없음'})"
        )
        prev_val_legacy = find_row_value_for_time(pool_rows, prev_time)
        if prev_val_legacy is None:
            return None
        prev_val = prev_val_legacy
    else:
        prev_val, _ = prev_hit

    if prev_val == 0:
        return None

    calc_rate = (current_val - prev_val) / prev_val * 100.0
    desc = (
        f"표에서 직접 계산: 현재값({cur_time or '?'}) {current_val} - "
        f"이전값({prev_time}) {prev_val} "
        f"→ 증가율 ({current_val}-{prev_val})/{prev_val}×100 = {calc_rate:.2f}%"
    )
    logger.info(f"[loop] {claim_id}: growth_rate 직접계산 성공 — {desc}")
    return (calc_rate, current_val, prev_val, desc)


def try_difference_from_rows(
    evidence: dict,
    schema,
    claim_id: str,
    all_fetch_observations: list | None = None,
) -> tuple[float, float, float, str] | None:
    """difference claim — 표 rows에서 current-prev 차이 직접 계산."""
    prev_time = getattr(schema, "prev_time_period", None) if schema else None
    if not prev_time:
        return None

    cur_time = getattr(schema, "time_period", None) if schema else None

    rows = list(evidence.get("rows") or [])
    pool_rows: list[dict] = []
    if all_fetch_observations:
        pool_rows = aggregate_rows_from_fetches(all_fetch_observations)
    for r in rows:
        if isinstance(r, dict) and r not in pool_rows:
            pool_rows.append(r)
    if not pool_rows:
        return None

    matched_row = evidence.get("matched_row") or {}
    criteria = extract_criteria_from_row(matched_row)

    current_val: float | None = None
    if cur_time:
        cur_hit = find_value_for_time_with_criteria(pool_rows, cur_time, criteria)
        if cur_hit is not None:
            current_val, _ = cur_hit
    if current_val is None:
        current_val = parse_row_dt(evidence.get("value"))
        if current_val is None and matched_row:
            current_val = parse_row_dt(matched_row.get("DT"))
    if current_val is None:
        return None

    prev_hit = find_value_for_time_with_criteria(pool_rows, prev_time, criteria)
    if prev_hit is None:
        logger.info(
            f"[loop] {claim_id}: difference 직접계산 — criteria 매칭 prev row "
            f"{prev_time!r} 못 찾음. 지표 무관 fallback "
            f"(pool={len(pool_rows)} rows, criteria={list(criteria.keys()) or '없음'})"
        )
        prev_val_legacy = find_row_value_for_time(pool_rows, prev_time)
        if prev_val_legacy is None:
            return None
        prev_val = prev_val_legacy
    else:
        prev_val, _ = prev_hit

    calc_diff = current_val - prev_val
    desc = (
        f"표에서 직접 계산: 현재값({cur_time or '?'}) {current_val} - "
        f"이전값({prev_time}) {prev_val} → 차이 {current_val}-{prev_val} = {calc_diff:.4f}"
    )
    logger.info(f"[loop] {claim_id}: difference 직접계산 성공 — {desc}")
    return (calc_diff, current_val, prev_val, desc)
