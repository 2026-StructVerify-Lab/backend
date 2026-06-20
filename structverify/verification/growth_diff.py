"""[리팩] 증가율/차이 자동 계산 — verifier._verify_growth_or_diff 분리 (fallback 프로필)"""
from __future__ import annotations

from collections.abc import Callable

from structverify.core.schemas import Claim, Evidence, MismatchType, VerificationResult
from structverify.utils.logger import get_logger

from .row_match import (
    extract_numeric_values,
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
