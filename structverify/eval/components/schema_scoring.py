"""Relaxed matching for component schema eval (fixture rows vs induced schema)."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def indicators_match(expected: str | None, actual: str | None, *, min_overlap: float = 0.5) -> bool:
    """Substring either way, or ≥min_overlap fraction of expected tokens."""
    if not (expected or "").strip():
        return True
    if not (actual or "").strip():
        return False
    e, a = expected.strip(), actual.strip()
    if e in a or a in e:
        return True
    et = {t for t in _TOKEN_RE.findall(e) if len(t) > 1}
    at = {t for t in _TOKEN_RE.findall(a) if len(t) > 1}
    if not et:
        return True
    overlap = len(et & at) / len(et)
    return overlap >= min_overlap


def time_periods_match(expected: str | None, actual: str | None) -> bool:
    """Match if same year or one normalized period contains the other (2022 ~ 202201)."""
    if not (expected or "").strip():
        return True
    if not (actual or "").strip():
        return False
    ed = re.sub(r"\D", "", expected or "")
    ad = re.sub(r"\D", "", actual or "")
    if not ed or not ad:
        return False
    if len(ed) >= 4 and len(ad) >= 4 and ed[:4] == ad[:4]:
        return True
    return ed.startswith(ad) or ad.startswith(ed)


def schema_values_match(
    expected: float,
    actual: float,
    *,
    expected_unit: str | None = None,
    actual_unit: str | None = None,
    rel: float = 0.1,
    abs_tol: float | None = None,
) -> bool:
    """Compare values with tolerance; try 만/천 scale factors when units differ."""
    if abs_tol is None:
        abs_tol = max(1.0, abs(expected) * 0.01)

    def _close(a: float, b: float) -> bool:
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom <= rel or abs(a - b) <= abs_tol

    multipliers: set[float] = {1.0}
    for u in (expected_unit or "", actual_unit or ""):
        ul = u.lower()
        if "만" in ul:
            multipliers.update((1e4, 1e-4))
        if "천" in ul and "천명개월" not in ul:
            multipliers.update((1e3, 1e-3))
        if "억" in ul:
            multipliers.update((1e8, 1e-8))

    pairs: list[tuple[float, float]] = []
    for me in multipliers:
        for ma in multipliers:
            pairs.append((expected * me, actual * ma))
    # Induced schema often omits scale in unit (e.g. 315.775 vs probe 3157750).
    for scale in (1e3, 1e4, 1e8):
        pairs.extend(
            [
                (expected, actual * scale),
                (expected * scale, actual),
            ]
        )
    return any(_close(e, a) for e, a in pairs)


def schema_field_score(
    *,
    indicator_ok: bool,
    value_ok: bool,
    time_ok: bool,
) -> float:
    return (float(indicator_ok) + float(value_ok) + float(time_ok)) / 3.0
