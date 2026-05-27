"""Verdict component eval scoring (strict vs production-aligned)."""
from __future__ import annotations


def relative_error(stated: float, official: float) -> float:
    denom = max(abs(stated), abs(official), 1e-9)
    return abs(stated - official) / denom


def verdict_strict_match(expected: str, predicted: str) -> bool:
    return expected == predicted


def verdict_aligned_match(
    expected: str,
    predicted: str,
    *,
    stated: float,
    official: float,
) -> bool:
    """Aligned with product: gray-zone unverifiable counts as mismatch signal."""
    if verdict_strict_match(expected, predicted):
        return True
    if expected == "mismatch" and predicted == "unverifiable":
        err = relative_error(stated, official)
        return 0.10 < err <= 0.30
    return False
