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


def _select_best_row(
    rows: list[dict],
    indicator: str | None,
    time_period: str | None,
    population: str | None = None,
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

    # 1차: indicator + 정확 시점 (+ population)
    for r in rows:
        if _ind_match(r) and _time_match(r) and _pop_match(r):
            return r

    # 2차: indicator + 정확 시점
    if prd_target:
        for r in rows:
            if _ind_match(r) and _time_match(r):
                return r

    # 3차: indicator만 매칭 — 가장 최근 시점 row 우선
    #   [v6.19] 단, claim이 월 단위면 연 단위 row는 제외 (가짜 매칭 방지)
    if ind_norm:
        matched = [
            r for r in rows if _ind_match(r) and _granularity_ok(r)
        ]
        if matched:
            matched.sort(
                key=lambda r: str(r.get("PRD_DE", "") or ""),
                reverse=True,
            )
            return matched[0]

    # 4차: 시점만 매칭
    if prd_target:
        for r in rows:
            if _time_match(r):
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

    def __init__(self, connector: Any = None, **kwargs: Any):
        """
        Args:
            connector: 기존 KOSISConnector 인스턴스. None이면 lazy 생성.
            **kwargs: KOSISConnector 생성 시 사용할 config (connector=None일 때만)
        """
        self._connector = connector
        self._connector_config = kwargs
        self._record_cache: dict[str, Any] = {}   # stat_id → StatRecord

        logger.info(
            f"[KOSISDataSource] 초기화. connector={'주입됨' if connector else 'lazy 생성 예정'}"
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
        tp = params.get("time_period")
        if tp and not params.get("prdSe"):
            prd_se, start, end = _parse_time_period(tp)
            fetch_params["prdSe"] = prd_se
            fetch_params["startPrdDe"] = start
            fetch_params["endPrdDe"] = end
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
        if relevance_query and not _is_table_relevant(relevance_query, stat_name_str):
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
                sample_keys = list((rows[0] or {}).keys())[:10]
                logger.warning(
                    f"[KOSISDataSource] indicator/time 매칭 row 못 찾음 "
                    f"(rows={len(rows)}, indicator={params.get('indicator')!r}, "
                    f"time={params.get('time_period')!r}) — evidence 없음 처리."
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