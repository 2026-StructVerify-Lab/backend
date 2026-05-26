"""Story anchor + compatible claim selection for single-narrative articles."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from structverify.eval.builder.schemas import ClaimSpec

_LOCAL_PATH_MARKERS = re.compile(
    r"지자체|광역시|특별자치|시기본|군기본|시사회|군사회|구기본"
)
_SOCIAL_SURVEY_MARKERS = re.compile(r"사회조사|인구\s*및\s*사회")
_REGION_SEGMENT_RE = re.compile(
    r"(?:^|>)\s*([^>]*?(?:특별자치도|광역시|특별시|도|시|군|구))\s*(?:>|$)"
)
_REGION_RAW_RE = re.compile(
    r"([가-힣]+(?:특별자치도|광역시|특별시|도|시|군|구))"
)
_PARENT_PREFIX_RE = re.compile(
    r"^([가-힣]+(?:특별자치도|광역시|특별시|도))"
)
_FINE_ADMIN_RE = re.compile(
    r"^[가-힣]{2,24}(?:특별자치도|광역시|특별시|시|군|구)$"
)
_REGION_SKIP = frozenset(
    {
        "인구",
        "사회",
        "경제",
        "보건",
        "교육",
        "환경",
        "지자체",
        "가구",
        "가족",
        "가족과가구",
        "혼인",
        "이혼",
        "출생",
        "사망",
        "농가",
        "인구수",
        "전국",
    }
)
_ADMIN_RANK = {
    "구": 4,
    "군": 4,
    "시": 3,
    "광역시": 5,
    "특별시": 5,
    "특별자치도": 5,
    "도": 1,
}


def normalize_region_token(token: str | None) -> str | None:
    """
    '경기도이천시' → '이천시', '전북특별자치도진안군' → '진안군' 등
    상위 행정구역이 접두로 붙은 KOSIS path 토큰을 정규화한다.
    """
    if not token:
        return None
    s = token.strip().replace(" ", "")
    if not s or s in _REGION_SKIP:
        return None
    m = _PARENT_PREFIX_RE.match(s)
    if m:
        rest = s[m.end() :]
        if rest and _FINE_ADMIN_RE.match(rest):
            return rest
    return s


def _admin_rank(token: str) -> int:
    for suffix, score in sorted(
        _ADMIN_RANK.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if token.endswith(suffix):
            return score
    return 0


def extract_region_token(category_path: str | None) -> str | None:
    """category_path에서 대표 지역 토큰 (시·군·구·광역시 우선, 도 단독은 후순위)."""
    if not category_path:
        return None
    best: tuple[int, str] | None = None
    for seg in category_path.split(">"):
        seg = seg.strip().replace(" ", "")
        if not seg or seg.startswith("MT_") or seg in _REGION_SKIP:
            continue
        if "가구" in seg and not seg.endswith(("시", "군", "구", "광역시")):
            continue
        for m in _REGION_RAW_RE.finditer(seg):
            token = normalize_region_token(m.group(1))
            if not token or token in _REGION_SKIP:
                continue
            r = _admin_rank(token)
            if r == 0:
                continue
            if best is None or r > best[0]:
                best = (r, token)
    return best[1] if best else None


def extract_year(time_period: str | None) -> int | None:
    if not time_period:
        return None
    m = re.search(r"(\d{4})", str(time_period))
    return int(m.group(1)) if m else None


def _spec_category_path(spec: ClaimSpec) -> str:
    if spec.gold_evidence and spec.gold_evidence.category_path:
        return spec.gold_evidence.category_path
    if spec.catalog_row:
        return str(spec.catalog_row.get("category_path") or "")
    return ""


def is_story_lock_region(token: str | None) -> bool:
    """
    기사 단일 지역 잠금에 쓸 수 있는 토큰인지.

    시·군·구·광역시(단일 광역)만 허용. '강원특별자치도'·'경기도' 같은 도 단위는
    여러 군이 한 풀에 섞이므로 제외한다.
    """
    if not token or token in _REGION_SKIP:
        return False
    if token.endswith(("광역시", "특별시")):
        return True
    if token.endswith(("시", "군", "구")):
        return not token.endswith("특별자치도")
    return False


def region_token_for_spec(spec: ClaimSpec) -> str | None:
    """기사 단일 지역 잠금·cross-region 검사용 대표 지역."""
    token = collect_fine_region_tokens(_spec_category_path(spec))
    if token and is_story_lock_region(token):
        return token
    return None


def story_region_tokens_from_specs(specs: list[ClaimSpec]) -> set[str]:
    tokens: set[str] = set()
    for spec in specs:
        token = region_token_for_spec(spec)
        if token:
            tokens.add(token)
    return tokens


def specs_single_region(
    specs: list[ClaimSpec],
    extra: ClaimSpec | None = None,
) -> bool:
    """스펙(＋선택 추가 스펙)이 최대 하나의 시·군·구·광역 단위만 쓰는지."""
    tokens = story_region_tokens_from_specs(specs)
    if extra:
        t = region_token_for_spec(extra)
        if t:
            tokens = tokens | {t}
    return len(tokens) <= 1


def region_compatible_with_locked_regions(
    category_path: str | None,
    locked_regions: set[str],
    *,
    allow_no_region: bool = True,
) -> bool:
    """이미 잠긴 지역 집합에 path가 맞는지 (전국·비지역 path는 allow_no_region 시 통과)."""
    if not locked_regions:
        return True
    if not category_path:
        return allow_no_region
    row = collect_fine_region_tokens(category_path) or normalize_region_token(
        extract_region_token(category_path)
    )
    if not row:
        return allow_no_region
    return row in locked_regions


def path_is_local(category_path: str | None) -> bool:
    if not category_path:
        return False
    if _LOCAL_PATH_MARKERS.search(category_path):
        return True
    if _SOCIAL_SURVEY_MARKERS.search(category_path):
        return True
    token = extract_region_token(category_path)
    if token and token.endswith(("시", "군", "구")):
        return True
    return False


def catalog_path_is_national(category_path: str | None) -> bool:
    """
    전국·비지역 KOSIS path (지자체/사회조사·시군구 lock 토큰 없음).
    """
    if not category_path or not str(category_path).strip():
        return False
    if path_is_local(category_path):
        return False
    if collect_fine_region_tokens(category_path):
        return False
    token = extract_region_token(category_path)
    if token and is_story_lock_region(token):
        return False
    return True


def catalog_row_is_national(row: dict[str, Any]) -> bool:
    return catalog_path_is_national(str(row.get("category_path") or ""))


def _locked_regions_from_anchor(
    anchor: StoryAnchor | None,
    locked_regions: set[str] | None = None,
) -> set[str]:
    locked = set(locked_regions or ())
    if anchor and anchor.region_token:
        norm = normalize_region_token(anchor.region_token)
        if norm:
            locked.add(norm)
    return locked


def region_compatible_with_anchor(
    anchor: StoryAnchor | None,
    category_path: str | None,
    *,
    single_region: bool = True,
    locked_regions: set[str] | None = None,
) -> bool:
    """앵커·잠긴 지역과 path가 단일 보도 범위인지."""
    if not single_region:
        return True
    locked = _locked_regions_from_anchor(anchor, locked_regions)
    return region_compatible_with_locked_regions(
        category_path, locked, allow_no_region=True
    )


@dataclass
class StoryAnchor:
    """First successful claim defines narrative scope for the article."""

    region_token: str | None = None
    anchor_year: int | None = None
    stat_id: str | None = None
    org_name: str | None = None
    survey_segment: str | None = None
    is_local: bool = False

    @classmethod
    def from_catalog_row(cls, row: dict[str, Any]) -> StoryAnchor:
        path = str(row.get("category_path") or "")
        parts = [p.strip() for p in path.split(">") if p.strip()]
        segment = ""
        if len(parts) >= 2:
            segment = parts[1] if parts[0].startswith("MT_") else parts[0]
        return cls(
            region_token=extract_region_token(path),
            anchor_year=None,
            stat_id=row.get("stat_id"),
            org_name=row.get("org_name") or "",
            survey_segment=segment or None,
            is_local=path_is_local(path),
        )

    @classmethod
    def from_spec(cls, spec: ClaimSpec) -> StoryAnchor:
        path = ""
        if spec.gold_evidence and spec.gold_evidence.category_path:
            path = spec.gold_evidence.category_path
        elif spec.catalog_row:
            path = str(spec.catalog_row.get("category_path") or "")

        year = extract_year(
            spec.gold_schema.time_period if spec.gold_schema else None
        )
        segment = ""
        if path:
            parts = [p.strip() for p in path.split(">") if p.strip()]
            if len(parts) >= 2:
                segment = parts[1] if parts[0].startswith("MT_") else parts[0]

        return cls(
            region_token=extract_region_token(path),
            anchor_year=year,
            stat_id=spec.gold_stat_id,
            org_name=(
                spec.gold_evidence.org_name if spec.gold_evidence else None
            )
            or (str(spec.catalog_row.get("org_name", "")) if spec.catalog_row else None),
            survey_segment=segment or None,
            is_local=path_is_local(path),
        )

    def narrative_hint(self) -> str:
        parts = []
        if self.region_token:
            parts.append(f"지역={self.region_token}")
        if self.anchor_year:
            parts.append(f"연도={self.anchor_year}")
        if self.survey_segment:
            parts.append(f"조사={self.survey_segment[:40]}")
        return ", ".join(parts) if parts else "동일 도메인 단일 보도"


def catalog_row_compatible(
    row: dict[str, Any],
    anchor: StoryAnchor | None,
    *,
    year_slack: int = 1,
    single_region: bool = True,
    locked_regions: set[str] | None = None,
) -> bool:
    if anchor is None and not locked_regions:
        return True

    path = str(row.get("category_path") or "")
    if not region_compatible_with_anchor(
        anchor,
        path,
        single_region=single_region,
        locked_regions=locked_regions,
    ):
        return False

    if anchor.survey_segment and path:
        seg_key = anchor.survey_segment.replace(" ", "")[:12]
        if len(seg_key) >= 4 and seg_key not in path.replace(" ", ""):
            if not anchor.stat_id or row.get("stat_id") != anchor.stat_id:
                return False

    return True


def spec_compatible_with_anchor(
    spec: ClaimSpec,
    anchor: StoryAnchor | None,
    *,
    year_slack: int = 1,
    single_region: bool = True,
    locked_regions: set[str] | None = None,
) -> bool:
    if anchor is None and not locked_regions:
        return True

    path = _spec_category_path(spec)
    locked = _locked_regions_from_anchor(anchor, locked_regions)

    if spec.intended_verdict == "unverifiable":
        if spec.unverifiable_recipe in ("U1", "U4", "U5"):
            if locked:
                return region_compatible_with_locked_regions(
                    path, locked, allow_no_region=True
                )
            return True
        if not path and spec.catalog_row:
            path = str(spec.catalog_row.get("category_path") or "")
        return region_compatible_with_anchor(
            anchor,
            path,
            single_region=single_region,
            locked_regions=locked_regions,
        )

    if not region_compatible_with_anchor(
        anchor,
        path,
        single_region=single_region,
        locked_regions=locked_regions,
    ):
        return False

    spec_year = extract_year(
        spec.gold_schema.time_period if spec.gold_schema else None
    )
    if anchor.anchor_year and spec_year:
        if abs(spec_year - anchor.anchor_year) > year_slack:
            return False

    if anchor.survey_segment and path:
        seg_key = anchor.survey_segment.replace(" ", "")[:12]
        if len(seg_key) >= 4 and seg_key not in path.replace(" ", ""):
            if not (
                spec.gold_stat_id
                and anchor.stat_id
                and spec.gold_stat_id == anchor.stat_id
            ):
                return False

    return True


def group_catalog_rows_by_region(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """catalog row를 대표 지역(시·군·구·광역)별로 묶는다."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path = str(row.get("category_path") or "")
        token = collect_fine_region_tokens(path) or normalize_region_token(
            extract_region_token(path)
        )
        if token and is_story_lock_region(token):
            grouped[token].append(row)
    return dict(grouped)


def pick_viable_region_pool(
    rows: list[dict[str, Any]],
    min_rows: int,
    rng: Any,
) -> tuple[str, list[dict[str, Any]]] | None:
    """min_rows 이상 catalog가 있는 지역 하나를 고른다."""
    grouped = group_catalog_rows_by_region(rows)
    viable = [(r, rs) for r, rs in grouped.items() if len(rs) >= min_rows]
    if not viable:
        return None
    region, pool = rng.choice(viable)
    return region, pool


def pick_viable_national_pool(
    rows: list[dict[str, Any]],
    min_rows: int,
    rng: Any,
) -> list[dict[str, Any]] | None:
    """min_rows 이상 전국(비지역) catalog row가 있으면 풀 반환."""
    national = [r for r in rows if catalog_row_is_national(r)]
    if len(national) < min_rows:
        return None
    pool = list(national)
    rng.shuffle(pool)
    return pool


def collect_fine_region_tokens(category_path: str | None) -> str | None:
    """기사 내 교차 지역 검사용 (정규화된 최하위 행정구역)."""
    if not category_path:
        return None
    token = extract_region_token(category_path)
    if not token:
        return None
    if _admin_rank(token) >= 3:
        return token
    return None


def article_claim_regions(claims: list[Any]) -> list[str]:
    """EvalClaim 리스트에서 대표 지역 토큰 수집."""
    regions: list[str] = []
    for claim in claims:
        ev = getattr(claim, "gold_evidence", None) or (
            claim.get("gold_evidence") if isinstance(claim, dict) else None
        )
        path = ""
        if ev:
            path = (
                ev.category_path
                if hasattr(ev, "category_path")
                else ev.get("category_path")
            ) or ""
        token = collect_fine_region_tokens(path)
        if token:
            regions.append(token)
    return regions
