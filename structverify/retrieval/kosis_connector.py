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
"""
retrieval/kosis_connector.py — KOSIS Open API 커넥터 (v3: CatalogSearchTool + LLM Agent)

[신준수 - 기존]
- KOSIS 통합검색 + getMeta(PRD/CMMT) 병렬 + Param fetch 구현

[김예슬 - 2026-04-30 / v4]
- search_and_fetch() 재설계:
  · 1단계: CatalogSearchTool.search() → 후보 stat_id top_k
  · 2단계: LLM Agent stat_id 선택 + 파라미터 결정 (HCX-DASH-002)
  · 3단계: KOSIS Param/statisticsParameterData.do fetch
  · 4단계: err=30 시 LLM Agent retry (최대 2회)

- fetch() 개선 (factcheck_test.py 참고):
  · prd_se 순회: Y → M → Q (연간 없으면 월간 시도)
  · objL 점진: err:20 시 objL2~8 순서로 추가
  · newEstPrdCnt=3 폴백: 기간 지정 실패 시 최신 3건
  · 단위 검증: is_same_unit_type으로 명↔개월 혼용 방지

[모듈 분리]
  CatalogSearchTool  ← catalog_search.py (pgvector 검색 전담)
  KOSISConnector     ← kosis_connector.py (LLM Agent + KOSIS API fetch)
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

_FETCH_MAX_RETRY = 2
_JSON_HEADERS: dict[str, str] = {
    "Accept":     "application/json, */*;q=0.1",
    "User-Agent": "StructVerify/1.0 (KOSIS OpenAPI; +https://kosis.kr/openapi/)",
}

# ── LLM Agent 프롬프트 ──────────────────────────────────────────────────────

_AGENT_SELECT_PROMPT = """당신은 한국 공식 통계 전문가입니다.
아래 검증 주장에 가장 적합한 KOSIS 통계표를 선택하고 조회 파라미터를 결정하세요.

검증 주장: "{claim_text}"
indicator: {indicator}
time_period: {time_period}
population: {population}

후보 통계표:
{candidates}

JSON으로만 답하세요:
{{
  "stat_id": "선택한 통계표 ID",
  "stat_name": "통계표명",
  "reason": "선택 이유 한 줄",
  "prd_se": "Y(연간)/M(월간)/Q(분기)",
  "start_prd_de": "시작 기간 (예: 2024, 202401)",
  "end_prd_de": "종료 기간"
}}"""

_AGENT_RETRY_PROMPT = """KOSIS API 조회가 실패했습니다. 파라미터를 수정하거나 다른 후보를 선택하세요.

검증 주장: "{claim_text}"
실패한 stat_id: {stat_id}
이전 파라미터: {prev_params}
오류: {error}

후보 통계표:
{candidates}

JSON으로만 답하세요:
{{
  "stat_id": "사용할 통계표 ID",
  "prd_se": "Y/M/Q",
  "start_prd_de": "기간",
  "end_prd_de": "기간",
  "reason": "수정 이유"
}}"""


# ── 단위 검증 유틸 (factcheck_test.py 참고) ──────────────────────────────────

def normalize_value(value: float, kosis_unit: str) -> float:
    """KOSIS 단위 → 기본 단위로 변환 (천명 → 명 등)"""
    u = kosis_unit.lower()
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
    try:
        r = await client.get(url, params=p, headers=_JSON_HEADERS, timeout=timeout)
        r.raise_for_status()
        data = _kosis_text_to_json(r.text or "")
        if data is None:
            return _meta_error_payload("parse")
        if isinstance(data, dict) and data.get("err") is not None and "row" not in data:
            return {"kosis_error": "api_err", "err": data.get("err"), "errMsg": data.get("errMsg")}
        return data
    except Exception as e:
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
        Catalog → LLM Agent → KOSIS fetch 파이프라인.

        [v1] KOSIS 통합검색 직접 → err=30 다수
        [v3] CatalogSearchTool(pgvector) → LLM Agent → fetch_with_retry
        """
        # 1단계: Catalog 검색 (pgvector + KOSIS API)
        candidates = await self.catalog.search(query, top_k=10)

        if not candidates:
            logger.warning(f"후보 없음: {query.keyword}")
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

        # 3단계: LLM Agent stat_id 선택
        agent_decision = await self._agent_select_stat(query, candidates)

        if agent_decision and agent_decision.get("stat_id"):
            selected_id  = agent_decision["stat_id"]
            prd_se       = agent_decision.get("prd_se", "Y")
            start_prd_de = agent_decision.get("start_prd_de", "")
            end_prd_de   = agent_decision.get("end_prd_de", "")
        else:
            # Agent 실패 → relevance_score 최고 후보
            best         = max(candidates, key=lambda r: r.relevance_score)
            selected_id  = best.stat_id
            prd_se       = "Y"
            start_prd_de = query.time_period or ""
            end_prd_de   = query.time_period or ""

        stat_rec = next((r for r in candidates if r.stat_id == selected_id), candidates[0])

        # 4단계: fetch (prd_se 순회 + objL 점진 + retry)
        last_error = ""
        for attempt in range(1, _FETCH_MAX_RETRY + 1):
            if attempt > 1:
                # LLM Agent retry — 파라미터 수정
                retry = await self._agent_retry_params(
                    query, candidates, selected_id,
                    {"prdSe": prd_se, "startPrdDe": start_prd_de, "endPrdDe": end_prd_de},
                    last_error,
                )
                if retry and retry.get("stat_id"):
                    selected_id  = retry["stat_id"]
                    stat_rec     = next((r for r in candidates if r.stat_id == selected_id), stat_rec)
                    prd_se       = retry.get("prd_se", "Y")
                    start_prd_de = retry.get("start_prd_de", "")
                    end_prd_de   = retry.get("end_prd_de", "")

            data = await self._fetch_with_retry(
                stat_id=selected_id,
                stat_rec=stat_rec,
                query=query,
                prd_se_hint=prd_se,
                start_prd_de=start_prd_de,
                end_prd_de=end_prd_de,
            )

            if data and data.official_value is not None:
                logger.info(
                    f"Evidence 조회 성공: [{selected_id}] "
                    f"value={data.official_value} {data.unit or ''}"
                )
                return data

            last_error = "데이터 없음"
            if data and data.raw_response:
                err = data.raw_response.get("err") or data.raw_response.get("error", "")
                last_error = f"err={err}"
                if data.raw_response.get("errMsg"):
                    last_error += f" {data.raw_response['errMsg']}"

            logger.warning(f"fetch 실패 (시도 {attempt}): {selected_id} | {last_error}")

        logger.warning(f"최종 Evidence 없음: {query.keyword}")
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

    async def search(self, query: ConnectorQuery) -> list[StatRecord]:
        """직접 호출 시 CatalogSearchTool로 위임"""
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