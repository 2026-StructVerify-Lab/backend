# 수정자: 신준수
# 수정 날짜: 2026-04-22
# 수정 내용: KOSIS API search() 실제 호출 구현
# 수정자: 신준수
# 수정 날짜: 2026-04-25
# 수정 내용: 통합검색 기본 5건 + getMeta(PRD, CMMT) 병렬 보강(통계표설명; fetch 재료)
# 수정자: 신준수
# 수정 날짜: 2026-04-26
# 수정 내용: Param/statisticsParameterData.do?method=getList (통계표선택) fetch 구현
# 수정자: 신준수
# 수정 날짜: 2026-04-27
# 수정 내용: fetch → StatData.official_value / unit / time_period (Param 셀 → 공용 필드)

# [DONE] KOSIS API search() 구현
# [DONE] getMeta(PRD/CMMT) 보강
# [DONE] KOSIS API fetch() Param/statisticsParameterData
# [TODO] 응답 파싱·obj/itm 매칭 정교화
# [2026-05-14 | 이수민] memory/v1: category_path 보강 로직 추가
#   - _lookup_category_path(): kosis_stat_catalog 테이블에서 stat_id로 직접 조회
#   - search_and_fetch 성공 시 stat_rec.metadata 우선, 없으면 DB 조회로 보강
#   - working memory 도메인 가드(verifier)에서 활용
"""
retrieval/kosis_connector.py — KOSIS Open API 커넥터 (v3: CatalogSearchTool + LLM Agent)

[신준수 - 기존]
- KOSIS 통합검색 + getMeta(PRD/CMMT) 병렬 + Param fetch 구현

[김예슬 - 2026-04-30 / v4]
- search_and_fetch() 재설계 (v4: 후보 순회 retry):
  · 1단계: CatalogSearchTool.search() → 후보 stat_id top_k
  · 2단계: LLM Agent stat_id 선택 + 파라미터 결정 (HCX-DASH-002)
  · 3단계: KOSIS Param/statisticsParameterData.do fetch
  · 4단계: 실패 → tried_ids에 추가 → Agent에게 "이건 실패, 남은 후보에서 골라라"
  · 5단계: 남은 후보에서 재선택 → fetch → 최대 _MAX_CANDIDATES(5)개 순회
  · Agent가 "NONE" 답하거나 남은 후보 없으면 포기

- fetch() 개선 (factcheck_test.py 참고):
  · prd_se 순회: Y → M → Q (연간 없으면 월간 시도)
  · objL 점진: err:20 시 objL2~8 순서로 추가
  · newEstPrdCnt=3 폴백: 기간 지정 실패 시 최신 3건
  · 단위 검증: is_same_unit_type으로 명↔개월 혼용 방지

[모듈 분리]
  CatalogSearchTool  ← catalog_search.py (pgvector 검색 전담)
  KOSISConnector     ← kosis_connector.py (LLM Agent + KOSIS API fetch)


[박재윤 - 2026-05-11]
- tried_log UnboundLocalError 버그 수정
  · candidates 없을 때 tried_ids, tried_log 초기화 누락 수정

[박재윤 - 2026-05-14 ]_is_table_relevant 국가 불일치 체크 추가
  · indicator에 외국 국가명 있고 테이블이 국내 통계면 → False
  · 미국 소비자물가 → 한국 소비자물가 테이블 매칭 방지

_is_table_relevant 해외 지역명 체크 추가
   · indicator에 국가명 없는데 테이블에 해외 지역명 있으면 → False
   · 합계출산율 → 합계출산율 남부·동남아시아 테이블 매칭 방지

_AGENT_SELECT_PROMPT 개선
   · 국가 일치 규칙 추가 (외국 지표 → 국내 테이블 선택 방지)
   · 지표/시점 일치 규칙 명시
   · 부적합 후보 NONE 반환 조건 추가


"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx
import json5  # type: ignore[import-untyped]

from structverify.core.schemas import GraphNode, GraphNodeType, ProvenanceRecord
from structverify.retrieval.base_connector import (
    BaseConnector, ConnectorQuery, StatData, StatRecord,
)
from structverify.retrieval.catalog_search import CatalogSearchTool
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

_FETCH_MAX_RETRY  = 2    # 단일 stat_id 내 prd_se 순회 retry
_MAX_CANDIDATES   = 5    # [v4] 후보 순회 최대 횟수
_JSON_HEADERS: dict[str, str] = {
    "Accept":     "application/json, */*;q=0.1",
    "User-Agent": "StructVerify/1.0 (KOSIS OpenAPI; +https://kosis.kr/openapi/)",
}

# ── LLM Agent 프롬프트 ──────────────────────────────────────────────────────
# [박재윤 - 2026-05-14]: _AGENT_SELECT_PROMPT 국가/지표/시점 일치 규칙 추가

_AGENT_SELECT_PROMPT = """당신은 한국 공식 통계 전문가입니다.
아래 검증 주장에 가장 적합한 KOSIS 통계표를 선택하고 조회 파라미터를 결정하세요.

검증 주장: "{claim_text}"
indicator: {indicator}
time_period: {time_period}
population: {population}

후보 통계표:
{candidates}

[선택 규칙 — 반드시 준수]
1. 국가 일치: indicator에 미국/일본/중국 등 외국이 명시되면 반드시 국제/해외 통계표 선택.
   한국 국내 통계표(예: 소비자물가등락률, 경제활동인구) 절대 선택 금지.
2. 지표 일치: indicator와 테이블명의 측정 대상이 같아야 함.
   예: indicator=출생아수 → 출생 관련 테이블, indicator=고용률 → 고용 관련 테이블.
3. 시점 일치: time_period가 "YYYY-MM"이면 prd_se=M(월간), "YYYY"이면 prd_se=Y(연간).
4. 부적합 후보: 위 조건 불만족 시 stat_id를 "NONE"으로 답하세요.

JSON으로만 답하세요:
{{
  "stat_id": "선택한 통계표 ID 또는 NONE",
  "stat_name": "통계표명",
  "reason": "선택 이유 한 줄 (국가/지표/시점 일치 여부 포함)",
  "prd_se": "Y(연간)/M(월간)/Q(분기)",
  "start_prd_de": "시작 기간 (예: 2024, 202401)",
  "end_prd_de": "종료 기간"
}}"""

_AGENT_RETRY_PROMPT = """KOSIS API 조회가 실패했습니다. 다른 후보 통계표를 선택하세요.

검증 주장: "{claim_text}"

이미 실패한 통계표 (다시 선택 금지):
{tried_list}

남은 후보 통계표:
{remaining_candidates}

[중요]
- 이미 실패한 stat_id는 절대 다시 선택하지 마세요
- 남은 후보 중에서 가장 적합한 통계표를 선택하세요
- 남은 후보가 모두 부적합하면 stat_id를 "NONE"으로 답하세요

JSON으로만 답하세요:
{{
  "stat_id": "선택한 통계표 ID 또는 NONE",
  "prd_se": "Y/M/Q",
  "start_prd_de": "기간",
  "end_prd_de": "기간",
  "reason": "선택 이유"
}}"""


# ── 단위 검증 유틸 (factcheck_test.py 참고) ──────────────────────────────────
# 이 함수를 kosis_connector.py의 기존 _is_table_relevant 대신 넣으세요

def _is_table_relevant(indicator: str, table_name: str) -> bool:
    """
    [v5] indicator와 테이블명이 같은 분야인지 키워드 검사.
    LLM 미사용 — 키워드 겹침 + 도메인 그룹 매칭 (빠름).

    "연평균기온" vs "기술개발 성과"      → False
    "연평균기온" vs "종관기상 평년값"    → True
    "평균 최저기온" vs "고온 일수 노출"  → True (기온 그룹)
    "시가총액" vs "산업별 취업자"        → False

    [박재윤 - 2026-05-14]: 국가 불일치 체크 추가
    indicator에 외국 국가명 있는데 테이블이 국내 통계면 → False
    """
    import re

    if not indicator or not table_name:
        return True

    ind_lower = indicator.lower()
    tbl_lower = table_name.lower()

    # [박재윤 - 2026-05-14]: 국가 불일치 체크
    _FOREIGN_COUNTRIES = {"미국", "일본", "중국", "독일", "영국", "유럽", "프랑스", "캐나다"}
    _FOREIGN_TABLE_MARKERS = {"국제", "해외", "외국", "세계"}
    foreign_in_indicator = any(c in ind_lower for c in _FOREIGN_COUNTRIES)
    foreign_in_table = any(c in tbl_lower for c in _FOREIGN_COUNTRIES | _FOREIGN_TABLE_MARKERS)
    if foreign_in_indicator and not foreign_in_table:
        return False

    # [박재윤 - 2026-05-14]: 해외 지역명 체크 추가
    # indicator에 국가명 없는데 테이블에 해외 지역명 있으면 → 해외 지역 통계 → False
    _REGION_MARKERS = {"남부·동남아시아", "동남아시아", "동북아시아", "중앙아시아",
                       "아프리카", "중동", "남미", "북미", "오세아니아", "서남아시아"}
    region_in_table = any(r in tbl_lower for r in _REGION_MARKERS)
    if not foreign_in_indicator and region_in_table:
        return False

    stopwords = {
        "통계", "현황", "조사", "결과", "항목", "지표", "전체", "기타",
        "관련", "변화", "추이", "비교", "분류", "지점별", "행정구역별",
        "성별", "연령별", "시도별", "시도", "연도별", "월별",
    }

    def _tokens(text: str) -> set[str]:
        words = set(re.split(r'[\s·,/()（）\-_]+', text.lower()))
        return {w for w in words if len(w) >= 2 and w not in stopwords}

    ind_tokens = _tokens(indicator)
    tbl_tokens = _tokens(table_name)

    # 1) 직접 키워드 겹침 (2글자 이상 토큰)
    overlap = ind_tokens & tbl_tokens
    if overlap:
        return True

    # 2) 부분 문자열 포함 (indicator 토큰이 테이블명에 포함)
    tbl_lower = table_name.lower()
    for tok in ind_tokens:
        if len(tok) >= 3 and tok in tbl_lower:
            return True

    # 3) 도메인 키워드 그룹 — 같은 그룹이면 관련 있음
    _GROUPS = [
        # 기상/기후
        {"기온", "기상", "기후", "날씨", "온도", "폭염", "한파", "평균기온",
         "연평균", "최저기온", "최고기온", "강수", "강우", "습도", "종관"},
        # 인구/가구
        {"인구", "출산", "사망", "고령", "청년", "세대", "가구", "출생", "혼인"},
        # 고용/노동
        {"고용", "실업", "취업", "임금", "근로", "노동", "쉬었음", "경제활동",
         "일자리", "실업률", "고용률"},
        # 경제/산업
        {"경제", "성장", "물가", "소비", "생산", "수출", "수입", "무역", "gdp"},
        # 주식/금융
        {"주가", "시총", "코스피", "코스닥", "시가총액", "금리", "환율", "주식"},
        # 교육
        {"교육", "학교", "학생", "진학", "졸업", "입학"},
        # 보건/의료
        {"의료", "건강", "질환", "사망률", "병원", "보건"},
        # 환경
        {"환경", "오염", "대기", "수질", "폐기물", "탄소", "에너지"},
        # 농업
        {"농업", "농가", "농림", "수확", "경작", "과수"},
        # 부동산/건설
        {"주택", "건설", "부동산", "아파트", "토지", "분양"},
    ]

    ind_lower = indicator.lower()
    for group in _GROUPS:
        ind_hit = any(kw in ind_lower for kw in group)
        tbl_hit = any(kw in tbl_lower for kw in group)
        if ind_hit and tbl_hit:
            return True

    return False

def normalize_value(value: float, kosis_unit: str) -> float:
    """KOSIS 단위 → 기본 단위로 변환 (천명 → 명 등)"""
    u = kosis_unit.lower()
    # [v4] 천명개월은 KOSIS 단위명 오류 — 실제로는 개월 단위, 변환 안 함
    if "천명개월" in u:
        return value
    if "천" in u:
        return value * 1000
    if "백만" in u or "million" in u:
        return value * 1_000_000
    if "억" in u:
        return value * 100_000_000
    return value


def is_same_unit_type(claim_unit: str, kosis_unit: str) -> bool:
    """단위 타입이 같은지 확인 (명↔개월 혼용 방지)"""
    claim_unit = (claim_unit or "").lower().strip()
    kosis_unit = (kosis_unit or "").lower().strip()

    if not claim_unit or not kosis_unit:
        return True

    # [v4] 천명개월은 KOSIS 단위명 오류 — 비교 자체를 통과
    if "천명개월" in kosis_unit:
        return True

    _TYPES = {
        "people":  ["명", "인구", "가구", "세대", "person"],
        "time":    ["개월", "월", "month", "년", "일", "주"],
        "ratio":   ["%", "퍼센트", "percent", "율", "비율"],
        "money":   ["원", "won", "달러", "dollar", "usd"],
    }

    def _get_type(u: str) -> str:
        for t, kws in _TYPES.items():
            if any(kw in u for kw in kws):
                return t
        return "unknown"

    ct = _get_type(claim_unit)
    kt = _get_type(kosis_unit)
    return (ct == "unknown" or kt == "unknown") or (ct == kt)


# ── [이수민 2026-05-14] category_path DB 직접 조회 헬퍼 ─────────────────────
# stat_rec.metadata에 category_path가 없는 경우 (KOSIS API 검색 결과 등)
# kosis_stat_catalog 테이블에서 직접 조회. 도메인 가드용.
async def _lookup_category_path(stat_id: str) -> str | None:
    try:
        import asyncpg
        dsn = os.environ.get(
            "PGVECTOR_DSN",
            "postgresql://postgres:1234@localhost:5432/factcheck",
        )
        conn = await asyncpg.connect(dsn)
        try:
            row = await conn.fetchrow(
                "SELECT category_path FROM kosis_stat_catalog WHERE stat_id = $1",
                stat_id,
            )
            return row["category_path"] if row else None
        finally:
            await conn.close()
    except Exception as e:
        logger.debug(f"category_path 조회 실패 ({stat_id}): {e}")
        return None


# ── 기존 헬퍼 (신준수) ───────────────────────────────────────────────────────

def _meta_error_payload(tag: str, exc: Exception | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"kosis_error": tag}
    if exc is not None:
        d["detail"] = str(exc)[:500]
    return d


def _kosis_text_to_json(text: str) -> Any | None:
    t = (text or "").strip()
    if not t or t.lstrip().startswith("<"):
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        try:
            return json5.loads(t)
        except (ValueError, TypeError):
            return None


def _kosis_cell_str(v: Any) -> str | None:
    s = str(v).strip() if v is not None else ""
    return s or None


def _rows_from_kosis_body(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and data.get("kosis_error"):
        return []
    if isinstance(data, dict) and isinstance(data.get("row"), list):
        return [x for x in data["row"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


async def kosis_get_meta(
    client: httpx.AsyncClient, base: str, api_key: str,
    org_id: str, tbl_id: str, meta_type: str, timeout: float,
) -> Any:
    p: dict[str, Any] = {
        "method": "getMeta", "type": meta_type, "apiKey": api_key,
        "orgId": org_id, "tblId": tbl_id, "format": "json", "content": "json",
    }
    if meta_type == "PRD":
        p["detail"] = "Y"
    url = f"{base.rstrip('/')}/statisticsData.do"
    logger.info("[ ] 요청: org_id=%s, tbl_id=%s, meta_type=%s", org_id, tbl_id, meta_type)
    try:
        r = await client.get(url, params=p, headers=_JSON_HEADERS, timeout=timeout)
        r.raise_for_status()
        logger.info("[kosis_get_meta] 응답 raw text (앞 500자): %s", (r.text or "")[:500])
        data = _kosis_text_to_json(r.text or "")
        if data is None:
            logger.info("[kosis_get_meta] JSON 파싱 실패 (data=None)")
            return _meta_error_payload("parse")
        if isinstance(data, dict) and data.get("err") is not None and "row" not in data:
            logger.info("[kosis_get_meta] API 오류 응답: err=%s, errMsg=%s", data.get("err"), data.get("errMsg"))
            return {"kosis_error": "api_err", "err": data.get("err"), "errMsg": data.get("errMsg")}
        logger.info("[kosis_get_meta] 파싱 성공: type=%s, len=%s", type(data).__name__, len(data) if isinstance(data, (list, dict)) else "-")
        return data
    except Exception as e:
        logger.info("[kosis_get_meta] HTTP 예외: %s", e)
        return _meta_error_payload("http", e)


async def kosis_enrich_stat_records(
    client: httpx.AsyncClient, base: str, api_key: str,
    records: list[StatRecord], *, timeout: float, record_concurrency: int = 5,
) -> None:
    if not records:
        return
    sem = asyncio.Semaphore(max(1, min(record_concurrency * 2, 16)))

    async def _one(rec: StatRecord) -> None:
        oid = (rec.org_id or (rec.metadata or {}).get("ORG_ID") or "")
        if isinstance(oid, str):
            oid = oid.strip() or None
        tid = (rec.stat_id or "").strip()
        if not oid or not tid:
            err = _meta_error_payload("no_org_or_tbl")
            rec.metadata["getMeta_PRD"] = err
            rec.metadata["getMeta_CMMT"] = err
            return
        try:
            async def _g(meta: str) -> Any:
                async with sem:
                    return await kosis_get_meta(client, base, api_key, oid, tid, meta, timeout)
            prd, cmmt = await asyncio.gather(_g("PRD"), _g("CMMT"))
        except Exception as e:
            epl = _meta_error_payload("enrich", e)
            prd, cmmt = epl, epl
        rec.metadata["getMeta_PRD"] = prd
        rec.metadata["getMeta_CMMT"] = cmmt

    await asyncio.gather(*[_one(r) for r in records])


# ══════════════════════════════════════════════════════════════════════════════

class KOSISConnector(BaseConnector):
    """
    KOSIS Open API 커넥터 (v3)

    search_and_fetch() 흐름:
      CatalogSearchTool → 후보 top_k
      LLM Agent → stat_id 선택 + 파라미터
      fetch_with_retry() → KOSIS API + prd_se 순회 + objL 점진
    """

    BASE_URL = "https://kosis.kr/openapi"

    def __init__(self, config: dict | None = None):
        self.config   = config or {}
        self.api_key  = os.environ.get(self.config.get("api_key_env", "KOSIS_API_KEY"), "")
        self.timeout  = self.config.get("timeout", 30)
        self.catalog  = CatalogSearchTool(config=self.config)

    # ── search_and_fetch (v3 핵심) ────────────────────────────────────────────

    async def search_and_fetch(self, query: ConnectorQuery) -> StatData | None:
        """
        Catalog → LLM Agent → KOSIS fetch 파이프라인 (v4: 후보 순회 retry).

        [v1] KOSIS 통합검색 직접 → err=30 다수
        [v3] CatalogSearchTool(pgvector) → LLM Agent → fetch_with_retry
        [v4 김예슬] 후보 순회 retry:
          1) Agent가 1순위 stat_id 선택 → fetch
          2) 실패 → tried_ids에 추가 → Agent에게 "이건 실패했다, 남은 후보에서 골라라"
          3) 남은 후보에서 재선택 → fetch
          4) 최대 _MAX_CANDIDATES(5)개까지 순회
          5) Agent가 "NONE" 답하거나 남은 후보 없으면 포기

        ReAct 패턴:
          Thought: "이 주장을 검증하려면 어떤 통계표가 필요한가?"
          Action:  stat_id 선택 + 파라미터 결정
          Observation: fetch 결과 (성공/실패)
          → 실패 시 Thought: "이 테이블은 안 됐다, 다른 후보에서 골라야 한다"
          → Action: 다음 stat_id 선택
        """
        # 1단계: Catalog 검색 (pgvector + KOSIS API)
        candidates = await self.catalog.search(query, top_k=10)

        if not candidates:
            logger.warning(f"후보 없음: {query.keyword}")
            # [v5] - 박재윤: tried_log 초기화 (candidates 없을 때 UnboundLocalError 수정)
            tried_ids: set[str] = set()
            tried_log: list[dict] = []
            # ── [v6] LLM Agent 검색어 재생성 ─────────────────────────────────
            simplified = await self._agent_simplify_keyword(query, tried_log)
            if simplified and simplified != (query.indicator or query.keyword or ""):
                logger.info(f"[재검색] Agent 검색어 변경: '{query.keyword}' → '{simplified}'")
                from structverify.retrieval.base_connector import ConnectorQuery
                retry_query = ConnectorQuery(
                    keyword=simplified,
                    indicator=simplified,
                    time_period=query.time_period,
                    population=query.population,
                    extra_params=query.extra_params,
                )
                retry_candidates = await self.catalog.search(retry_query, top_k=5)
                retry_candidates = [c for c in retry_candidates if c.stat_id not in tried_ids]

                for rc in retry_candidates[:3]:
                    if not _is_table_relevant(simplified, rc.stat_name):
                        continue
                    data = await self._fetch_with_retry(
                        stat_id=rc.stat_id, stat_rec=rc, query=query,
                        prd_se_hint="Y",
                        start_prd_de=query.time_period or "",
                        end_prd_de=query.time_period or "",
                    )
                    if data and data.official_value is not None:
                        logger.info(f"[재검색] 성공: [{rc.stat_id}] {rc.stat_name}")
                        return data

            logger.warning(f"최종 Evidence 없음 ({len(tried_ids)}개 시도): {query.keyword}")
            return None

        # 2단계: getMeta 보강 (PRD/CMMT)
        if self.api_key and self.config.get("enrich_get_meta", True):
            try:
                base = (self.config.get("base_url") or self.BASE_URL).rstrip("/")
                async with httpx.AsyncClient(timeout=self.timeout) as mclient:
                    await kosis_enrich_stat_records(
                        mclient, base, self.api_key, candidates[:5],
                        timeout=float(self.timeout),
                    )
            except Exception as e:
                logger.debug(f"getMeta 보강 실패: {e}")

        # ── [v4] 후보 순회 retry 루프 ────────────────────────────────────
        tried_ids: set[str] = set()
        tried_log: list[dict] = []   # Agent에게 보여줄 실패 이력

        for round_idx in range(_MAX_CANDIDATES):
            remaining = [c for c in candidates if c.stat_id not in tried_ids]
            if not remaining:
                logger.info("모든 후보 소진 → 종료")
                break

            # 3단계: stat_id 선택
            if round_idx == 0:
                # 첫 번째: LLM Agent가 최적 후보 선택
                agent_decision = await self._agent_select_stat(query, candidates)
            else:
                # [v4]: Agent 재선택 대신 순차 순회 (factcheck_test.py 방식)
                # Agent가 재선택해도 엉뚱한 테이블을 고르는 경우가 많아서
                # relevance_score 순서대로 순차 시도하는 게 정확도 더 높음
                agent_decision = {
                    "stat_id":      remaining[0].stat_id,
                    "prd_se":       "Y",
                    "start_prd_de": query.time_period or "",
                    "end_prd_de":   query.time_period or "",
                }
                logger.info(
                    f"[순차 순회] 다음 후보: [{remaining[0].stat_id}] "
                    f"{remaining[0].stat_name}"
                )

            if agent_decision and agent_decision.get("stat_id"):
                selected_id = agent_decision["stat_id"]
                # "NONE" 답변 → Agent가 적합한 후보 없다고 판단
                if selected_id.upper() == "NONE":
                    logger.info("Agent: 적합한 후보 없음 → 종료")
                    break
                # 이미 시도한 stat_id를 다시 선택한 경우 → 강제로 다음 후보
                if selected_id in tried_ids:
                    selected_id = remaining[0].stat_id
                prd_se       = agent_decision.get("prd_se", "Y")
                start_prd_de = agent_decision.get("start_prd_de", "")
                end_prd_de   = agent_decision.get("end_prd_de", "")
            else:
                # Agent 실패 → relevance_score 최고 후보 (시도 안 한 것 중)
                selected_id  = remaining[0].stat_id
                prd_se       = "Y"
                start_prd_de = query.time_period or ""
                end_prd_de   = query.time_period or ""

            stat_rec = next(
                (r for r in candidates if r.stat_id == selected_id),
                remaining[0]
            )
            tried_ids.add(selected_id)

            logger.info(
                f"[후보 {round_idx+1}/{_MAX_CANDIDATES}] "
                f"시도: [{selected_id}] {stat_rec.stat_name}"
            )

            # 4단계: fetch (prd_se 순회 + objL 점진)
            data = await self._fetch_with_retry(
                stat_id=selected_id,
                stat_rec=stat_rec,
                query=query,
                prd_se_hint=prd_se,
                start_prd_de=start_prd_de,
                end_prd_de=end_prd_de,
            )

            if data and data.official_value is not None:
                # [v5] 가짜 match 방지 — 테이블 관련성 체크
                indicator = query.indicator or query.keyword or ""
                table_name = stat_rec.stat_name or ""
                if not _is_table_relevant(indicator, table_name):
                    logger.warning(
                        f"테이블 관련성 없음 → skip: [{selected_id}] "
                        f"{table_name} vs indicator={indicator}"
                    )
                    tried_ids.add(selected_id)
                    tried_log.append({
                        "stat_id":   selected_id,
                        "stat_name": table_name,
                        "error":     "테이블 관련성 없음",
                    })
                    continue

                # [이수민 2026-05-14] working memory 도메인 가드용
                # stat_rec.metadata에 있으면 사용, 없으면 DB 직접 조회로 보강
                if data.category_path is None:
                    data.category_path = (stat_rec.metadata or {}).get("category_path")
                if data.category_path is None:
                    data.category_path = await _lookup_category_path(selected_id)

                logger.info(
                    f"Evidence 조회 성공 (후보 {round_idx+1}): [{selected_id}] "
                    f"value={data.official_value} {data.unit or ''} "
                    f"category={data.category_path}"
                )
                return data

            # 실패 이력 기록 → 다음 라운드에서 Agent에게 전달
            last_error = "데이터 없음"
            if data and data.raw_response:
                err = data.raw_response.get("err") or data.raw_response.get("error", "")
                last_error = f"err={err}"
                if data.raw_response.get("errMsg"):
                    last_error += f" {data.raw_response['errMsg']}"

            tried_log.append({
                "stat_id":   selected_id,
                "stat_name": stat_rec.stat_name,
                "error":     last_error,
            })
            logger.warning(
                f"fetch 실패 (후보 {round_idx+1}): [{selected_id}] "
                f"{stat_rec.stat_name} | {last_error}"
            )

        logger.warning(f"최종 Evidence 없음 ({len(tried_ids)}개 시도): {query.keyword}")
        return None

    # ── prd_se 순회 + objL 점진 fetch (factcheck_test.py 참고) ───────────────

    async def _fetch_with_retry(
        self,
        stat_id: str,
        stat_rec: StatRecord,
        query: ConnectorQuery,
        prd_se_hint: str = "Y",
        start_prd_de: str = "",
        end_prd_de: str = "",
    ) -> StatData | None:
        """
        prd_se 순회(Y→M→Q) + objL 점진 + newEstPrdCnt 폴백.

        factcheck_test.py의 fetch_kosis_data() 로직을 비동기로 재구현.
        """
        # 연도 추출
        time_ref = query.time_period or ""
        year_m = re.search(r"(\d{4})", start_prd_de or time_ref)
        year = year_m.group(1) if year_m else "2024"

        # prd_se 순회 전략
        prd_strategies = [
            {"prdSe": prd_se_hint, "startPrdDe": start_prd_de or year,
             "endPrdDe": end_prd_de or year},
        ]
        # hint가 Y가 아니면 Y도 시도
        if prd_se_hint != "Y":
            prd_strategies.insert(0, {"prdSe": "Y", "startPrdDe": year, "endPrdDe": year})
        # M, Q 추가
        for prd, sp, ep in [("M", f"{year}01", f"{year}12"), ("Q", f"{year}01", f"{year}04")]:
            if prd != prd_se_hint:
                prd_strategies.append({"prdSe": prd, "startPrdDe": sp, "endPrdDe": ep})

        # 기간 지정 실패 시 최신 데이터 폴백
        fallbacks = [
            {"prdSe": p, "newEstPrdCnt": "3"}
            for p in ["Y", "M", "Q"]
        ]

        org_id = (
            stat_rec.org_id
            or (stat_rec.metadata or {}).get("ORG_ID")
            or ""
        )
        if not org_id:
            logger.debug(f"org_id 없음: {stat_id}")
            return None

        prd_m  = (stat_rec.metadata or {}).get("getMeta_PRD")
        cmmt_m = (stat_rec.metadata or {}).get("getMeta_CMMT")
        prd_rows  = _rows_from_kosis_body(prd_m)
        cmmt_rows = _rows_from_kosis_body(cmmt_m)

        obj_l1 = "ALL"
        itm_id = "ALL"
        if cmmt_rows:
            r0 = cmmt_rows[0]
            obj_l1 = str(r0.get("OBJ_ID") or r0.get("C1") or "ALL").strip() or "ALL"
            itm_id = str(r0.get("ITM_ID") or "ALL").strip() or "ALL"

        base = (self.config.get("base_url") or self.BASE_URL).rstrip("/")

        for strategy in prd_strategies + fallbacks:
            base_params: dict[str, Any] = {
                "method":  "getList",
                "apiKey":  self.api_key,
                "format":  "json",
                "content": "json",
                "orgId":   org_id,
                "tblId":   stat_id,
                "objL1":   obj_l1,
                "itmId":   itm_id,
                "prdSe":   strategy["prdSe"],
            }
            if "startPrdDe" in strategy:
                base_params["startPrdDe"] = strategy["startPrdDe"]
                base_params["endPrdDe"]   = strategy.get("endPrdDe", strategy["startPrdDe"])
            if "newEstPrdCnt" in strategy:
                base_params["newEstPrdCnt"] = strategy["newEstPrdCnt"]

            data = await self._try_with_objl_escalation(base, base_params, stat_id, stat_rec)
            if data is not None:
                return data

        return None

    async def _try_with_objl_escalation(
        self,
        base: str,
        base_params: dict[str, Any],
        stat_id: str,
        stat_rec: StatRecord,
    ) -> StatData | None:
        """
        objL 점진 추가 (err:20 → objL2~8 순서로 추가).
        factcheck_test.py의 _try_with_objL_escalation 참고.
        """
        result = await self._call_kosis_param(base, base_params, stat_id, stat_rec)
        if result is None:
            return None

        # err:20 = objL 부족 → 점진 추가
        if result.raw_response.get("err") == "20":
            for level in range(2, 9):
                key = f"objL{level}"
                if key not in base_params:
                    base_params[key] = "ALL"
                    result = await self._call_kosis_param(base, base_params, stat_id, stat_rec)
                    if result is None:
                        return None
                    if result.raw_response.get("err") != "20":
                        break

        # 여전히 에러면 None
        if result.raw_response.get("err"):
            return None

        return result if result.official_value is not None else None

    async def _call_kosis_param(
        self,
        base: str,
        params: dict[str, Any],
        stat_id: str,
        stat_rec: StatRecord,
    ) -> StatData | None:
        """KOSIS Param/statisticsParameterData.do 단일 호출"""
        public_req = {k: v for k, v in params.items() if k != "apiKey"}
        tnm = getattr(stat_rec, "stat_name", stat_id) or stat_id

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(
                    f"{base}/Param/statisticsParameterData.do",
                    params=params,
                    headers=_JSON_HEADERS,
                )
                r.raise_for_status()
                text = (r.text or "").strip()

                if not text or text.lstrip().startswith("<"):
                    return StatData(stat_id=stat_id, stat_name=tnm, values={},
                                    raw_response={"error": "empty_or_html", "request": public_req})

                j = _kosis_text_to_json(text)
                if j is None:
                    return StatData(stat_id=stat_id, stat_name=tnm, values={},
                                    raw_response={"error": "json_parse", "request": public_req})

                if isinstance(j, dict) and j.get("err") is not None and "row" not in j:
                    return StatData(stat_id=stat_id, stat_name=tnm, values={},
                                    raw_response={
                                        "err": j.get("err"),
                                        "errMsg": j.get("errMsg"),
                                        "request": public_req,
                                    })

                drows = _rows_from_kosis_body(j)
                if not drows:
                    return StatData(stat_id=stat_id, stat_name=tnm, values={},
                                    raw_response={**(j if isinstance(j, dict) else {}),
                                                  "request": public_req})

                cell0 = drows[0]
                tnm2  = (cell0.get("TBL_NM") or tnm) or stat_id
                raw_out = {**j, "request": public_req} if isinstance(j, dict) else {
                    "row": drows, "request": public_req}

                dt_s = str(cell0.get("DT") or "").strip()
                val: float | None = None
                if dt_s:
                    try:
                        val = float(dt_s.replace(",", ""))
                    except ValueError:
                        pass

                # 단위 타입 검증
                kosis_unit = _kosis_cell_str(cell0.get("UNIT_NM")) or ""
                # verifier.py에서 unit 타입 확인용으로 metadata에 저장
                tp = _kosis_cell_str(cell0.get("PRD_DE"))

                return StatData(
                    stat_id=stat_id,
                    stat_name=str(tnm2).strip() or stat_id,
                    values={
                        "value":   val,
                        "DT":      cell0.get("DT"),
                        "PRD_DE":  cell0.get("PRD_DE"),
                        "ITM_NM":  cell0.get("ITM_NM"),
                        "UNIT_NM": kosis_unit,
                    },
                    raw_response=raw_out,
                    official_value=val,
                    unit=kosis_unit,
                    time_period=tp,
                )

        except httpx.HTTPError as e:
            logger.error("KOSIS param HTTP: %s", e)
            return None
        except Exception as e:
            logger.error("KOSIS param: %s", e)
            return None

    # ── LLM Agent ────────────────────────────────────────────────────────────

    async def _agent_select_stat(
        self, query: ConnectorQuery, candidates: list[StatRecord]
    ) -> dict[str, Any] | None:
        """HCX-DASH-002로 후보 중 최적 stat_id 선택"""
        if not candidates:
            return None
        try:
            from structverify.utils.llm_client import LLMClient
            llm = LLMClient(config=self.config.get("llm", {}))
        except ImportError:
            return None

        candidate_text = "\n".join([
            f"  {i+1}. [{r.stat_id}] {r.stat_name} ({r.org_name}) "
            f"[{r.metadata.get('category_path','')}]"
            for i, r in enumerate(candidates[:10])
        ])

        raw_claim = (query.extra_params or {}).get("raw_claim", "")
        prompt = _AGENT_SELECT_PROMPT.format(
            claim_text=raw_claim[:200] or query.keyword,
            indicator=query.indicator or "",
            time_period=query.time_period or "",
            population=query.population or "",
            candidates=candidate_text,
        )
        logger.info(
            "[LLM 진입 직전 | _agent_select_stat] claim=%r | indicator=%r | time=%r | population=%r | 후보(%d개):\n%s",
            raw_claim[:200] or query.keyword,
            query.indicator or "",
            query.time_period or "",
            query.population or "",
            len(candidates),
            candidate_text,
        )
        try:
            result = await llm.generate_json(
                prompt=prompt,
                system_prompt="한국 통계 전문가. JSON으로만 답하세요.",
                model_tier="light",
            )
            if result and result.get("stat_id"):
                logger.info(
                    f"Agent 선택: [{result['stat_id']}] {result.get('stat_name','')} "
                    f"— {result.get('reason','')}"
                )
            return result
        except Exception as e:
            logger.debug(f"Agent 선택 실패: {e}")
            return None

    async def _agent_retry_params(
        self,
        query: ConnectorQuery,
        candidates: list[StatRecord],
        prev_stat_id: str,
        prev_params: dict,
        error: str,
    ) -> dict[str, Any] | None:
        """HCX-DASH-002로 fetch 실패 시 파라미터 수정"""
        try:
            from structverify.utils.llm_client import LLMClient
            llm = LLMClient(config=self.config.get("llm", {}))
        except ImportError:
            return None

        candidate_text = "\n".join([
            f"  {i+1}. [{r.stat_id}] {r.stat_name} ({r.org_name})"
            for i, r in enumerate(candidates[:10])
        ])
        raw_claim = (query.extra_params or {}).get("raw_claim", "")
        prompt = _AGENT_RETRY_PROMPT.format(
            claim_text=raw_claim[:200] or query.keyword,
            stat_id=prev_stat_id,
            prev_params=json.dumps(prev_params, ensure_ascii=False)[:300],
            error=error[:200],
            candidates=candidate_text,
        )
        logger.info(
            "[LLM 진입 직전 | _agent_retry_params] claim=%r | 실패 stat_id=%s | prev_params=%s | error=%r | 후보(%d개):\n%s",
            raw_claim[:200] or query.keyword,
            prev_stat_id,
            json.dumps(prev_params, ensure_ascii=False)[:300],
            error[:200],
            len(candidates),
            candidate_text,
        )
        try:
            result = await llm.generate_json(
                prompt=prompt,
                system_prompt="한국 통계 전문가. JSON으로만 답하세요.",
                model_tier="light",
            )
            return result
        except Exception as e:
            logger.debug(f"Agent retry 실패: {e}")
            return None

    # ── 기존 search() 유지 (catalog_search 폴백용) ───────────────────────────

    async def _agent_retry_with_rotation(
        self,
        query: ConnectorQuery,
        remaining: list[StatRecord],
        tried_log: list[dict],
    ) -> dict[str, Any] | None:
        """
        [v4 김예슬] 실패한 후보 제외하고 남은 후보에서 재선택.

        ReAct Observation:
          "DT_1BC0501 → 데이터 없음"
          "DT_705003 → err=30"
        → Thought: "산업 취업자 통계는 안 됨, 경제활동인구 통계로 시도"
        → Action: 다음 stat_id 선택
        """
        if not remaining:
            return None

        try:
            from structverify.utils.llm_client import LLMClient
            llm = LLMClient(config=self.config.get("llm", {}))
        except ImportError:
            return None

        # 실패 이력 텍스트
        tried_text = "\n".join([
            f"  - [{t['stat_id']}] {t['stat_name']} → {t['error']}"
            for t in tried_log
        ]) or "  (없음)"

        # 남은 후보 텍스트
        remaining_text = "\n".join([
            f"  {i+1}. [{r.stat_id}] {r.stat_name} ({r.org_name}) "
            f"[{r.metadata.get('category_path', '')}]"
            for i, r in enumerate(remaining[:10])
        ])

        raw_claim = (query.extra_params or {}).get("raw_claim", "")
        prompt = _AGENT_RETRY_PROMPT.format(
            claim_text=raw_claim[:200] or query.keyword,
            tried_list=tried_text,
            remaining_candidates=remaining_text,
        )
        logger.info(
            "[LLM 진입 직전 | _agent_retry_with_rotation] claim=%r | 실패 이력(%d개):\n%s\n남은 후보(%d개):\n%s",
            raw_claim[:200] or query.keyword,
            len(tried_log),
            tried_text,
            len(remaining),
            remaining_text,
        )

        try:
            result = await llm.generate_json(
                prompt=prompt,
                system_prompt="한국 통계 전문가. 이미 실패한 통계표는 절대 다시 선택하지 마세요. JSON으로만 답하세요.",
                model_tier="light",
            )
            if result and result.get("stat_id"):
                logger.info(
                    f"Agent 재선택: [{result['stat_id']}] "
                    f"{result.get('reason', '')}"
                )
            return result
        except Exception as e:
            logger.debug(f"Agent 재선택 실패: {e}")
            return None


    async def _agent_simplify_keyword(
            self, query: ConnectorQuery, tried_log: list[dict]
        ) -> str | None:
            """LLM Agent가 실패 이력을 보고 검색어를 재생성"""
            try:
                from structverify.utils.llm_client import LLMClient
                llm = LLMClient(config=self.config.get("llm", {}))
            except ImportError:
                return None

            tried_text = "\n".join([
                f"  - {t['stat_name']} → {t['error']}"
                for t in tried_log
            ]) or "  (없음)"

            raw_claim = (query.extra_params or {}).get("raw_claim", "")
            prompt = f"""KOSIS 통계표 검색이 실패했습니다. 검색어를 바꿔서 재시도하려 합니다.

    검증 주장: "{raw_claim[:200]}"
    기존 검색어: "{query.keyword}"
    실패한 테이블들:
    {tried_text}

    원래 검색어로는 관련 테이블을 찾지 못했습니다.
    KOSIS 통계표 이름에 실제로 들어갈 법한 더 단순하고 핵심적인 검색어 1~2단어를 제안하세요.

    예시:
    "연평균 기온 순위" → "기온"
    "청년 쉬었음 비율 변화" → "경제활동인구"
    "평균 최저기온 및 평균 최고기온" → "기온"

    검색어만 답하세요 (설명 없이):"""

            try:
                result = await llm.generate(
                    prompt=prompt,
                    system_prompt="KOSIS 통계 검색 전문가. 검색어만 답하세요.",
                    model_tier="light",
                )
                keyword = result.strip().strip('"\'')
                if keyword and len(keyword) >= 2:
                    return keyword
            except Exception as e:
                logger.debug(f"Agent 검색어 재생성 실패: {e}")
            return None
    async def search(self, query: ConnectorQuery) -> list[StatRecord]:
        return await self.catalog.search(query)

    async def fetch(self, stat_id: str, params: dict[str, Any]) -> StatData:
        """BaseConnector 인터페이스 유지 (search_and_fetch 내부에서 직접 사용)"""
        stat_rec = params.get("stat_record") or StatRecord(
            stat_id=stat_id, stat_name=stat_id
        )
        query = params.get("query") or ConnectorQuery(keyword=stat_id)
        prd_se = params.get("prdSe", "Y")
        sp     = params.get("startPrdDe", "")
        ep     = params.get("endPrdDe", "")

        result = await self._fetch_with_retry(
            stat_id=stat_id,
            stat_rec=stat_rec,
            query=query,
            prd_se_hint=prd_se,
            start_prd_de=sp,
            end_prd_de=ep,
        )
        return result or StatData(
            stat_id=stat_id, stat_name=stat_id, values={},
            raw_response={"error": "fetch_failed"},
        )

    def to_graph_nodes(self, data: StatData) -> list[GraphNode]:
        return [GraphNode(
            node_id=f"evidence:{data.stat_id}",
            node_type=GraphNodeType.EVIDENCE,
            label=data.stat_name,
            properties=data.values,
        )]

    def tag_provenance(self, data: StatData, query: ConnectorQuery) -> ProvenanceRecord:
        return ProvenanceRecord(
            source_connector="KOSIS",
            source_id=data.stat_id,
            query_used=query.keyword,
            raw_snapshot=data.raw_response,
        )