"""
verification/combiners.py — Multi-evidence combiner (v6.3 신규)

[배경]
v6.2까지는 evidence 1개 ↔ claim 1:1 비교만 가능.
하지만 실제 claim 다수가 두 시점의 measurement를 결합해야 검증 가능:
  - "올 4월 6.7% 증가"  : (current - baseline) / baseline * 100 = ?
  - "0.04명 증가"        : current - baseline = ?
  - "2.3도 웃돌았다"     : current - baseline = ?

[v6.3 설계]
schema_inductor가 claim의 value_role에 따라 EvidencePlan을 생성:
  - measurement: requirements=[primary]                   → combiner=direct
  - delta:       requirements=[endpoint_a, endpoint_b]    → combiner=delta
  - ratio:       requirements=[endpoint_a, endpoint_b]    → combiner=ratio_pct

verifier는 이 enum에 따라 적절한 combiner 함수를 호출하여
evidence 들의 official_value를 결합한 computed_value를 만든 뒤
claim.value와 비교.

[설계 원칙]
- combiner는 결정론적 함수 (LLM 호출 없음)
- evidence가 부족하면 None 반환 → verifier가 unverifiable 처리
- LLM이 임의 수식을 만들 수 없게 enum으로 제한 (보안성)
"""
from __future__ import annotations

from typing import Sequence

from structverify.core.schemas import Evidence
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# combiner 이름 enum (str)
COMBINER_DIRECT = "direct"
COMBINER_DELTA = "delta"
COMBINER_RATIO_PCT = "ratio_pct"
VALID_COMBINERS = {COMBINER_DIRECT, COMBINER_DELTA, COMBINER_RATIO_PCT}


def find_endpoint(evidences: Sequence[Evidence], role: str) -> Evidence | None:
    """role 매칭하는 evidence 찾기. 없으면 None."""
    for e in evidences:
        if e.requirement_role == role:
            return e
    return None


def combine(
    combiner: str,
    evidences: Sequence[Evidence],
) -> tuple[float | None, str]:
    """
    evidences를 combiner로 결합하여 (computed_value, formula_str) 반환.

    Returns:
        (computed_value, human_readable_formula)
        - 결합 실패 시 (None, "사유")

    Raises:
        ValueError: 알 수 없는 combiner
    """
    if combiner not in VALID_COMBINERS:
        raise ValueError(f"unknown combiner: {combiner}")

    if combiner == COMBINER_DIRECT:
        return _combine_direct(evidences)
    if combiner == COMBINER_DELTA:
        return _combine_delta(evidences)
    if combiner == COMBINER_RATIO_PCT:
        return _combine_ratio_pct(evidences)

    return None, "no combiner matched"


# ── direct: evidence 1개 그대로 ──────────────────────────────────────────

def _combine_direct(evidences: Sequence[Evidence]) -> tuple[float | None, str]:
    """role='primary' 또는 첫 evidence의 official_value 그대로."""
    primary = find_endpoint(evidences, "primary")
    if primary is None and evidences:
        primary = evidences[0]
    if primary is None or primary.official_value is None:
        return None, "primary evidence 없음"
    return float(primary.official_value), f"primary={primary.official_value}"


# ── delta: a − b ─────────────────────────────────────────────────────────

def _combine_delta(evidences: Sequence[Evidence]) -> tuple[float | None, str]:
    """endpoint_a − endpoint_b (양쪽 모두 measurement)."""
    a = find_endpoint(evidences, "endpoint_a")
    b = find_endpoint(evidences, "endpoint_b")
    if a is None or b is None:
        missing = []
        if a is None: missing.append("endpoint_a")
        if b is None: missing.append("endpoint_b")
        return None, f"missing: {','.join(missing)}"
    if a.official_value is None or b.official_value is None:
        return None, "endpoint value None"

    av, bv = float(a.official_value), float(b.official_value)
    delta = av - bv
    formula = f"({av} − {bv}) = {delta:+.4g}"
    return delta, formula


# ── ratio_pct: (a − b) / b × 100 ────────────────────────────────────────

def _combine_ratio_pct(evidences: Sequence[Evidence]) -> tuple[float | None, str]:
    """
    증가율(%) = (current - baseline) / baseline * 100
    endpoint_a = current, endpoint_b = baseline 가정.

    baseline이 0이면 0 division → 실패 처리.
    """
    a = find_endpoint(evidences, "endpoint_a")
    b = find_endpoint(evidences, "endpoint_b")
    if a is None or b is None:
        missing = []
        if a is None: missing.append("endpoint_a")
        if b is None: missing.append("endpoint_b")
        return None, f"missing: {','.join(missing)}"
    if a.official_value is None or b.official_value is None:
        return None, "endpoint value None"

    av, bv = float(a.official_value), float(b.official_value)
    if abs(bv) < 1e-9:
        return None, "baseline=0 (zero division)"

    ratio_pct = (av - bv) / bv * 100.0
    formula = f"({av} − {bv}) / {bv} × 100 = {ratio_pct:+.3f}%"
    return ratio_pct, formula