"""Text/value helpers for eval prose and validation."""
from __future__ import annotations

import re
from typing import Any

_MD_BOLD_RE = re.compile(r"\*\*\s*([^*]+?)\s*\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_INLINE_BACKTICK_RE = re.compile(r"`([^`]+)`")


def normalize_claim_text(text: str) -> str:
    """Whitespace-normalize claim text for harness matching (validator parity)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def strip_llm_markdown_emphasis(text: str) -> str:
    """LLM **bold** / *italic* / `code` 제거 — 실제 뉴스 평문 형식."""
    if not text:
        return text
    return "\n".join(_strip_emphasis_line(line) for line in text.split("\n"))


def _strip_emphasis_line(line: str) -> str:
    s = line
    prev = None
    while prev != s:
        prev = s
        s = _MD_BOLD_RE.sub(r"\1", s)
    s = s.replace("**", "")
    s = _MD_ITALIC_RE.sub(r"\1", s)
    s = _INLINE_BACKTICK_RE.sub(r"\1", s)
    return s


def prose_has_markdown_emphasis(text: str) -> bool:
    """파이프라인 입력에 남으면 안 되는 LLM 마크다운 강조."""
    if not text:
        return False
    if "**" in text:
        return True
    return bool(_MD_ITALIC_RE.search(text) or _INLINE_BACKTICK_RE.search(text))


def normalize_kosis_unit(unit: str | None) -> str | None:
    """KOSIS UNIT_NM 노이즈 정리 (백만원%, 개% 등)."""
    if unit is None:
        return None
    u = str(unit).strip()
    if not u:
        return None
    replacements = (
        ("백만원%", "백만원"),
        ("만원%", "만원"),
        ("개 %", "개"),
        ("개%", "개"),
        ("명%", "명"),
        ("건%", "건"),
        ("%p", "%p"),
    )
    for old, new in replacements:
        u = u.replace(old, new)
    return u or None


def format_gold_number(value: float, *, max_decimals: int = 4) -> str:
    """claim/validator 매칭용 표기 (콤마·소수)."""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    rounded = round(value, max_decimals)
    s = f"{rounded:,.4f}".rstrip("0").rstrip(".")
    parts = s.split(".")
    if len(parts) == 2:
        return f"{parts[0]},{parts[1]}"
    return s


def format_value_with_unit(value: float, unit: str | None) -> str:
    num = format_gold_number(value)
    u = normalize_kosis_unit(unit) or ""
    if u == "%" or u == "%p":
        if 0 <= value <= 1 and abs(value) < 1:
            pct = value * 100 if u == "%" else value
            if abs(pct - round(pct)) < 1e-6:
                return f"{int(round(pct))}%"
            return f"{pct:g}%"
        return f"{num}%"
    if u:
        return f"{num}{u}"
    return num


def _numeric_variants(value: float) -> list[str]:
    variants = set()
    variants.add(format_gold_number(value))
    variants.add(f"{value:g}")
    if abs(value - round(value)) < 1e-9:
        variants.add(str(int(round(value))))
    compact = format_gold_number(value).replace(",", "")
    variants.add(compact)
    if 0 < value <= 1:
        pct = int(round(value * 100))
        variants.add(str(pct))
        variants.add(f"{pct}%")
        variants.add(f"{value * 100:g}%")
    return [v for v in variants if v]


def claim_text_reflects_gold_value(
    claim_text: str,
    value: float | None,
    unit: str | None,
) -> bool:
    """claim_text에 gold_schema.value가 실질적으로 포함됐는지."""
    if value is None:
        return True
    text = claim_text.replace(",", "").replace(" ", "")
    for variant in _numeric_variants(float(value)):
        v = variant.replace(",", "").replace(" ", "").replace("%", "")
        if v and v in text:
            return True
    formatted = format_value_with_unit(float(value), unit)
    core = formatted.replace(",", "").replace(" ", "")
    if core and core.replace("%", "") in text:
        return True
    return False


def _news_scope_prefix(population: str | None) -> str:
    pop = (population or "전체").strip()
    if pop in ("전국", "전체", "국가", "national"):
        return "전국 "
    if pop and pop not in ("전체",):
        return f"{pop} "
    return ""


def _short_indicator(indicator: str | None) -> str:
    ind = (indicator or "지표").strip()
    if len(ind) <= 48:
        return ind
    if "-" in ind:
        tail = ind.split("-")[-1].strip()
        if tail:
            return tail[:48]
    return ind[:48]


_BANNED_CLAIM_PHRASING_RE = re.compile(
    r"할\s*수\s*있|있을\s*수|확정되지|할\s*것으로|될\s*것으로"
)
_FORECAST_CLAIM_RE = re.compile(r"전망|예상|예측|목표")
_MALFORMED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3}){2,},\d")


def claim_has_banned_phrasing(text: str) -> bool:
    """탐지(check-worthy)에서 자주 false가 되는 표현."""
    if not text:
        return False
    return bool(_BANNED_CLAIM_PHRASING_RE.search(text) or _FORECAST_CLAIM_RE.search(text))


def prose_has_malformed_number(text: str) -> bool:
    """56,353,75 같은 깨진 수치 표기."""
    return bool(text and _MALFORMED_NUMBER_RE.search(text))


def lead_contains_gold_values(lead_text: str, claims: list[Any]) -> bool:
    """리드에 gold 수치가 포함됐는지 (claim 전용 문단 제외한 텍스트용)."""
    if not lead_text.strip():
        return False
    for claim in claims:
        verdict = getattr(claim, "gold_verdict", None) or (
            claim.get("gold_verdict") if isinstance(claim, dict) else None
        )
        if verdict not in ("match", "mismatch"):
            continue
        sch = getattr(claim, "gold_schema", None) or (
            claim.get("gold_schema") if isinstance(claim, dict) else None
        )
        if not sch:
            continue
        value = getattr(sch, "value", None)
        if value is None and isinstance(sch, dict):
            value = sch.get("value")
        unit = getattr(sch, "unit", None)
        if unit is None and isinstance(sch, dict):
            unit = sch.get("unit")
        if value is not None and claim_text_reflects_gold_value(
            lead_text, float(value), unit
        ):
            return True
    return False


# 뉴스 단정형 마감 — claim_id 해시로 골고루 섞음 (단일 템플릿 homogenization 방지)
_NEWS_CLAIM_PATTERNS: tuple[str, ...] = (
    "{tp} {scope}{ind}은(는) {val}으로 집계됐다.",
    "{tp} {scope}{ind}은(는) {val}로 나타났다.",
    "{tp} {scope}{ind}은(는) {val}으로 확인됐다.",
    "{tp} {scope}{ind}은(는) {val}으로 보고됐다.",
    "{tp} {scope}{ind}은 {val}이다.",
    "{tp} 기준 {scope}{ind} 수치는 {val}이다.",
)


def claim_variant_index(claim_id: str, n: int = len(_NEWS_CLAIM_PATTERNS)) -> int:
    """Deterministic per-claim template rotation (not identical across dataset)."""
    if not claim_id or n <= 0:
        return 0
    h = sum(ord(c) for c in claim_id)
    return h % n


def build_claim_text_from_spec(
    spec: Any,
    *,
    style: str = "caption",
    variant_index: int | None = None,
) -> str:
    """LLM 실패·교정용 결정론적 claim 문장 (gold value/unit/time 반영).

    style:
      - caption: 통계표 캡션체 (…은(는) …로 나타났다) — 레거시 eval
      - news: 뉴스 단정형 (…으로 집계됐다) — 단일 패턴
      - news_varied: 뉴스 단정형 — claim_id 기반 패턴 로테이션 (fallback용)
    """
    sch = spec.gold_schema
    if spec.intended_verdict == "unverifiable":
        if spec.unverifiable_recipe == "U1":
            tp = (sch.time_period if sch else None) or "향후"
            return f"{tp} 해당 분야 전망은 공식 통계만으로 단정하기 어렵다."
        if spec.unverifiable_recipe == "U5":
            return f"최근 {sch.indicator if sch else '지표'}에 대한 논의가 이어지고 있다."
        return f"{sch.indicator if sch else '관련 지표'}에 대한 주장이 제기됐다."

    if not sch or sch.value is None:
        return f"{sch.indicator if sch else '통계'} 관련 보도가 이어지고 있다."

    val_s = format_value_with_unit(float(sch.value), sch.unit)
    tp = sch.time_period or "해당 연도"
    ind = _short_indicator(sch.indicator)

    if style in ("news", "news_varied"):
        scope = _news_scope_prefix(sch.population)
        if style == "news":
            return f"{tp} {scope}{ind}은(는) {val_s}으로 집계됐다."
        idx = variant_index
        if idx is None:
            idx = claim_variant_index(getattr(spec, "claim_id", "") or "")
        pattern = _NEWS_CLAIM_PATTERNS[idx % len(_NEWS_CLAIM_PATTERNS)]
        return pattern.format(tp=tp, scope=scope, ind=ind, val=val_s)

    return f"{tp} {ind}은(는) {val_s}로 나타났다."
