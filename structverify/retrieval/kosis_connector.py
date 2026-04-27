"""
retrieval/kosis_connector.py — KOSIS Open API 커넥터 (과제 핵심)

[참고] KOSIS Open API — https://kosis.kr/openapi/index/index.jsp
  getList(통계표 목록 검색), getStatData(상세 데이터 조회) 엔드포인트 사용

- 담당자: 신준수
"""
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

from __future__ import annotations
import asyncio
import json
import os
import re
from typing import Any
import httpx
import json5  # type: ignore[import-untyped]  # PyPI json5, Apache-2.0; 비RFC 응답
from structverify.core.schemas import GraphNode, GraphNodeType, ProvenanceRecord
from structverify.retrieval.base_connector import (
    BaseConnector, ConnectorQuery, StatData, StatRecord)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# 통합검색 resultCount 기본값(한 페이지에 가져올 통계표 후보 개수; search_result_count로 덮기)
_DEFAULT_SEARCH_COUNT = 5
# getMeta/통계자료 GET 공통 헤더(일부 환경에서 User-Agent·Accept 없으면 HTML만 오는 경우 대비)
_JSON_HEADERS: dict[str, str] = {
    "Accept": "application/json, */*;q=0.1",
    "User-Agent": "StructVerify/1.0 (KOSIS OpenAPI; +https://kosis.kr/openapi/)",
}


# 역할: getMeta/Param 호출이 실패했을 때 metadata에 꽂을 작은 에러 dict 만들기
def _meta_error_payload(tag: str, exc: Exception | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"kosis_error": tag}
    if exc is not None:
        d["detail"] = str(exc)[:500]
    return d


# 역할: HTTP 응답 본문 → JSON (통합검색·getMeta 응답 파싱 공용)
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


# 역할: KOSIS 셀 문자열 → StatData unit/time_period
def _kosis_cell_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


# 역할: getMeta/Param JSON 본문(dict)에서 row[] 꺼내기 (에러-only dict면 [])
def _rows_from_kosis_body(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and data.get("kosis_error"):
        return []
    if isinstance(data, dict) and isinstance(data.get("row"), list):
        return [x for x in data["row"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


# 역할: 통계표설명 getMeta 1회 호출. 060103(PRD)·060105(CMMT) 는 동일 URL(statisticsData.do),
#       `meta_type` 이 PRD면 가이드 060103(주기/시점), CMMT면 060105(주석+분류·항목 id 힌트)
async def kosis_get_meta(
    client: httpx.AsyncClient,
    base: str,
    api_key: str,
    org_id: str,
    tbl_id: str,
    meta_type: str,
    timeout: float,
) -> Any:
    p: dict[str, Any] = {
        "method": "getMeta",
        "type": meta_type,
        "apiKey": api_key,
        "orgId": org_id,
        "tblId": tbl_id,
        "format": "json",
        "content": "json",
    }
    if meta_type == "PRD":
        p["detail"] = "Y"  # PRD: 전체 시점 정보(가이드 선택)
    url = f"{base.rstrip('/')}/statisticsData.do"
    try:
        r = await client.get(url, params=p, headers=_JSON_HEADERS, timeout=timeout)
        r.raise_for_status()
        data = _kosis_text_to_json(r.text or "")
        if data is None:
            return _meta_error_payload("parse")
        if (
            isinstance(data, dict)
            and data.get("err") is not None
            and "row" not in data
        ):
            return {
                "kosis_error": "api_err",
                "err": data.get("err"),
                "errMsg": data.get("errMsg"),
            }
        return data
    except Exception as e:  # noqa: BLE001
        logger.debug("kosis getMeta %s %s/%s: %s", meta_type, org_id, tbl_id, e)
        return _meta_error_payload("http", e)


# 역할: 통계 후보 1행(StatRecord)에 대해 060103+060105 를 동시에 때림(gather) → metadata 키에 저장
async def _kosis_enrich_one(
    client: httpx.AsyncClient,
    base: str,
    api_key: str,
    rec: StatRecord,
    sem: asyncio.Semaphore,
    timeout: float,
) -> None:
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
        # PRD·CMMT 각각 kosis_get_meta(060103/060105) — sem 으로 동시에 뜨는 getMeta 개수 제한
        async def _g(meta: str) -> Any:
            async with sem:
                return await kosis_get_meta(
                    client, base, api_key, oid, tid, meta, timeout,
                )

        prd, cmmt = await asyncio.gather(_g("PRD"), _g("CMMT"))
    except Exception as e:  # noqa: BLE001
        logger.warning("kosis getMeta gather: %s", e)
        epl = _meta_error_payload("enrich", e)
        prd, cmmt = epl, epl
    rec.metadata["getMeta_PRD"] = prd
    rec.metadata["getMeta_CMMT"] = cmmt


# 역할: 통합검색으로 나온 "모든" 후보 행에 대해 _kosis_enrich_one 을 병렬 스케줄 + 세마포로 동시성 관리
async def kosis_enrich_stat_records(
    client: httpx.AsyncClient,
    base: str,
    api_key: str,
    records: list[StatRecord],
    *,
    timeout: float,
    record_concurrency: int = 5,
) -> None:
    if not records:
        return
    sem = asyncio.Semaphore(max(1, min(record_concurrency * 2, 16)))

    async def _one(rec: StatRecord) -> None:
        await _kosis_enrich_one(client, base, api_key, rec, sem, timeout)

    await asyncio.gather(*[_one(r) for r in records])


class KOSISConnector(BaseConnector):
    """KOSIS Open API 커넥터"""
    BASE_URL = "https://kosis.kr/openapi"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.api_key = os.environ.get(self.config.get("api_key_env", "KOSIS_API_KEY"), "")
        self.timeout = self.config.get("timeout", 30)

    async def search(self, query: ConnectorQuery) -> list[StatRecord]:
        # [v0]
        # """
        # KOSIS 통계표 목록 검색 (getList)
        # API 엔드포인트: GET /openapi/getList
        # params: apiKey, format=json, vwCd=MT_ZTITLE, searchNm=query.keyword
        # Args: query: 검색 쿼리 (keyword 필수)
        # Returns: StatRecord 리스트 (통계표 ID, 이름, 기관명, 가용 기간 등)
        # """
        """
        KOSIS 통계표 통합검색 (statisticsSearch.do, method=getList)

        API 엔드포인트: GET <base_url>/statisticsSearch.do
        params: method=getList, apiKey, searchNm=query.keyword, format=json, sort=RANK,
            resultCount, startCount, (선택) orgId=query.extra_params

        Args:
            query: 검색 쿼리 (keyword 필수)

        Returns:
            StatRecord 리스트 (표 ID, 이름, 기관, 기간, relevance_score; 실패 시 [])
        """
        # [v0] 초기 stub
        # logger.warning(f"KOSIS search stub: {query.keyword}")
        # return [StatRecord(stat_id="DT_1EA1019", stat_name="경영주 연령별 농가",
        #                    org_name="통계청", available_periods=["2024"],
        #                    relevance_score=0.95)]

        # [v1] - 실제 HTTP: 통계표만 가져옴(셀 수치는 `fetch`에서)
        if not self.api_key:
            logger.error("KOSIS API 키가 설정되지 않았습니다. 환경변수 KOSIS_API_KEY를 확인하세요.")
            return []
        if not (query.keyword or "").strip():
            # searchNm(검색어)이 비어 있으면 KOSIS에도 쓸 수 없고, `StatRecord`도 의미 없음
            logger.warning("KOSIS 통합검색: keyword가 비어 있습니다.")
            return []

        # `base_url` 은 config로 바꾸기 쉬우라고(테스트/미러 대비). 없으면 클래스 상수
        base = (self.config.get("base_url") or self.BASE_URL).rstrip("/")  # KOSIS OpenAPI 루트 URL
        # 한 번에 몇 개까지 가져올지: 나중에 `search_and_fetch`가 relevance 높은 걸 골라도, 후보 풀은 이 개수(기본 5)
        n_max = int(
            self.config.get("search_result_count", _DEFAULT_SEARCH_COUNT),
        )  # resultCount(한 페이지 상한)
        extra = query.extra_params or {}  # orgId, content 덮어쓰기 등

        # KOSIS "통합검색" 파라미터(가이드: statisticsSearch.do, method=getList)
        # - searchNm: 사용자 키워드 → `ConnectorQuery.keyword`
        # - format, content(헤더 유형 html|json) 등
        # - orgId: (선택) extra_params
        request_params: dict[str, Any] = {
            "method": "getList",
            "apiKey": self.api_key,
            "searchNm": query.keyword.strip(),
            "format": "json",
            "content": str(extra.get("content") or "json"),  # 가이드: 헤더 유형 — json 쪽 권장
            "sort": "RANK",  # 관련도순(위에 올수록 더 맞는 표라고 설명되어 있음)
            "resultCount": str(n_max),
            "startCount": "1",
        }
        if extra.get("orgId"):
            request_params["orgId"] = str(extra["orgId"])  # 기관으로 검색 제한(선택)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 일부 환경에서 User-Agent/Accept 없으면 HTML(안내/차단)만 오는 사례가 있어 JSON 요청에 맞게 둔다.
                r = await client.get(  # httpx 응답(상태줄·헤더·본문)
                    f"{base}/statisticsSearch.do",
                    params=request_params,
                    headers={
                        "Accept": "application/json, */*;q=0.1",
                        "User-Agent": "StructVerify/1.0 (KOSIS OpenAPI; +https://kosis.kr/openapi/)",
                    },
                )
                r.raise_for_status()  # 4xx/5xx면 예외
                text = (r.text or "").strip()  # 응답 본문 문자열
                ct = (r.headers.get("content-type") or "").lower()  # Content-Type(로그·판별용)
                if not text:
                    logger.error("KOSIS 응답 본문이 비어 있음 (status=%s content-type=%s)", r.status_code, ct)
                    return []
                # text/html 이어도 본문이 API 데이터일 수 있어 content-type에 "html"이 들어갔다고 HTML로 치지 않는다.
                if text.lstrip().startswith("<"):
                    logger.error(
                        "KOSIS 응답이 마크업(HTML)로 보임(키/URL/서버 오류 HTML 등). status=%s content-type=%s head=%.400s",
                        r.status_code,
                        ct,
                        text,
                    )
                    return []
                try:
                    data = json.loads(text)  # 표준 JSON이면 dict 또는 list
                except json.JSONDecodeError as e:
                    m = re.search(  # {err, errMsg} 비표준 문자열(키 무따옴표) 대비
                        r'err\s*:\s*"(\d+)"\s*,\s*errMsg\s*:\s*"([^"]*)"',
                        text,
                    )
                    if m:
                        logger.error(
                            "KOSIS API 오류(비표준 응답): err=%s errMsg=%s (키·파라미터·발급 상태 확인)",
                            m.group(1),
                            m.group(2),
                        )
                        return []
                    try:
                        data = json5.loads(text)  # KOSIS식 느슨한 JSON
                    except (ValueError, TypeError) as e2:
                        logger.error(
                            "KOSIS 본문 파싱 실패(json·json5): %s / %s | content-type=%s | 앞200자=%.200r",
                            e,
                            e2,
                            ct,
                            text,
                        )
                        return []
        except httpx.HTTPStatusError as e:
            logger.error(f"KOSIS API HTTP 에러: {e.response.status_code} - {e.response.text[:500]}")
            return []
        except httpx.TimeoutException:
            logger.error(f"KOSIS API 타임아웃 ({self.timeout}초 초과)")
            return []
        except Exception as e:
            logger.error(f"KOSIS API 호출 실패: {e}")
            return []

        if (
            isinstance(data, dict)
            and data.get("err") is not None
            and "row" not in data
        ):
            logger.error(
                "KOSIS API 오류(파싱 후 dict): err=%s errMsg=%s",
                data.get("err"),
                data.get("errMsg"),
            )
            return []

        # JSON을 "행" 리스트로 맞춤: 통합검색 응답이 주로 { "row": [ {...}, ... ] } 형태
        if isinstance(data, dict) and isinstance(data.get("row"), list):
            rows = [x for x in data["row"] if isinstance(x, dict)]  # 래퍼 { row: [...] }
        elif isinstance(data, list):
            rows = [x for x in data if isinstance(x, dict)]  # 본문이 곧배열 [...]
        else:
            rows = []  # 예상 밖이면 빈 리스트(크래시보다 `[]` 반환이 상위 흐름에 안전)

        n = len(rows)  # KOSIS가 돌려준 행 수(필터 전)
        records: list[StatRecord] = []
        for i, item in enumerate(rows):
            tid = (item.get("TBL_ID") or "").strip()  # 통계표 ID(없으면 스킵)
            if not tid:
                continue
            tnm = (item.get("TBL_NM") or "").strip() or tid  # 통계표명
            org = item.get("ORG_NM")  # 기관명(없을 수 있음)
            org_id = (item.get("ORG_ID") or "").strip() or None  # 기관코드(통계자료 fetch orgId)
            # 기간 필드(있으면): 후보에 "어느 시점 자료인지" 힌트
            periods: list[str] = []
            for k in ("STRT_PRD_DE", "END_PRD_DE"):
                v = item.get(k)  # 수록기간 시작/끝(가이드 필드명)
                if v is not None and str(v).strip() and str(v).strip() not in periods:
                    periods.append(str(v).strip())
            # relevance_score: KOSIS가 RANK로 이미 정렬이므로, 앞 index일수록 점수 높게(1.0 → 내려감)
            # `BaseConnector.search_and_fetch` 는 max(relevance_score)로 하나 고름
            rel = 1.0 if n <= 1 else max(0.05, 1.0 - (i / (n - 1)) * 0.95)  # 순위→상대 점수 -> kosis 응답이 rank 순으로 정렬되어 있어 임시로 정렬해봄
            records.append(StatRecord(
                stat_id=tid, stat_name=tnm, org_name=org, org_id=org_id,
                available_periods=periods, relevance_score=rel,
                metadata=dict(item),
            ))
        if not records:
            return []
        # 통계표설명 getMeta(주기+주석/분류·항목 힌트). 통합검색 Http 클라이언트가 닫힌 뒤 별도 클라이언트(동일 base).
        # 끄기: config enrich_get_meta=False. 동시 행 수: meta_record_concurrency(기본 5, 세마포=대략 2×이 값).
        if self.config.get("enrich_get_meta", True) and (self.api_key or "").strip():
            try:
                b = (self.config.get("base_url") or self.BASE_URL).rstrip("/")
                async with httpx.AsyncClient(timeout=self.timeout) as mclient:
                    await kosis_enrich_stat_records(
                        mclient,
                        b,
                        self.api_key,
                        records,
                        timeout=float(self.timeout),
                        record_concurrency=int(
                            self.config.get("meta_record_concurrency", 5),
                        ),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "getMeta PRD+CMMT 보강 실패(검색 StatRecord 는 유지): %s", e,
                )
        logger.info(f"KOSIS API 검색+메타(옵션) 완료: {len(records)}개 (응답 {n}행)")
        return records

    async def fetch(self, stat_id: str, params: dict[str, Any]) -> StatData:
        # [v0] - stub (고정 수치)
        # logger.warning(f"KOSIS fetch stub: {stat_id}")
        # return StatData(stat_id=stat_id, stat_name="경영주 연령별 농가",
        #                 values={"total": 166558, "age_65_plus": 106877, "ratio": 64.2})

        # 목적: 통계자료 API 통계표선택( devGuide_0201 ) — Param/statisticsParameterData.do?method=getList
        cq: ConnectorQuery | None = params.get("query")
        if not isinstance(cq, ConnectorQuery):
            cq = None
        rec: StatRecord | None = params.get("stat_record")
        extra: dict[str, Any] = dict(cq.extra_params) if cq else {}
        mdef: dict[str, Any] = dict(self.config.get("fetch_defaults", {}))
        mdef.update(extra)

        tnm = (getattr(rec, "stat_name", None) or stat_id) if rec else stat_id
        tnm = str(tnm).strip() or stat_id

        if not (self.api_key or "").strip():
            logger.error("KOSIS fetch: API 키 없음")
            return StatData(
                stat_id=stat_id,
                stat_name=tnm,
                values={},
                raw_response={"error": "no_api_key"},
            )

        org_id: str | None = None
        if mdef.get("orgId") is not None and str(mdef.get("orgId", "")).strip() != "":
            org_id = str(mdef["orgId"]).strip()
        elif rec is not None and rec.org_id:
            org_id = str(rec.org_id).strip()
        if not org_id and rec is not None:
            o = (rec.metadata or {}).get("ORG_ID")
            if o is not None and str(o).strip() != "":
                org_id = str(o).strip()
        if not org_id:
            logger.error("KOSIS fetch: orgId 없음 (StatRecord·extra_params·metadata.ORG_ID)")
            return StatData(
                stat_id=stat_id,
                stat_name=tnm,
                values={},
                raw_response={"error": "missing_orgId", "tblId": stat_id},
            )

        prd_m = (rec.metadata or {}).get("getMeta_PRD") if rec else None
        cmmt_m = (rec.metadata or {}).get("getMeta_CMMT") if rec else None
        prd_rows = _rows_from_kosis_body(prd_m)
        cmmt_rows = _rows_from_kosis_body(cmmt_m)

        obj_l1: str = str(mdef.get("objL1") or "").strip()
        itm_id: str = str(mdef.get("itmId") or "").strip()
        if cmmt_rows and (not obj_l1 or not itm_id):
            r0 = cmmt_rows[0]
            if not obj_l1:
                obj_l1 = str(
                    r0.get("OBJ_ID") or r0.get("C1") or "ALL",
                ).strip() or "ALL"
            if not itm_id:
                itm_id = str(r0.get("ITM_ID") or "ALL").strip() or "ALL"
        if not obj_l1:
            obj_l1 = "ALL"
        if not itm_id:
            itm_id = "ALL"

        prd_se = str(mdef.get("prdSe") or "").strip()
        if not prd_se and prd_rows:
            prd_se = str(prd_rows[0].get("PRD_SE") or "").strip() or "Y"
        if not prd_se and cq and (cq.time_period or "").strip():
            td = re.sub(r"\D", "", cq.time_period or "")
            if len(td) == 4:
                prd_se = "Y"
            elif len(td) == 6:
                prd_se = "M"
            elif len(td) == 8:
                prd_se = "D"
        if not prd_se:
            prd_se = "Y"

        sp = str(mdef.get("startPrdDe") or "").strip()
        ep = str(mdef.get("endPrdDe") or "").strip()
        if not sp and not ep and cq and (cq.time_period or "").strip():
            td = re.sub(r"\D", "", cq.time_period)
            if len(td) in (4, 6, 8):
                sp, ep = td, td
        if not sp and not ep and prd_rows:
            pde = (prd_rows[0].get("PRD_DE") or "")
            pde = str(pde).strip()
            if pde:
                sp, ep = pde, pde

        base = (self.config.get("base_url") or self.BASE_URL).rstrip("/")
        ncnt = str(mdef.get("newEstPrdCnt") or "").strip()
        req: dict[str, Any] = {
            "method": "getList",
            "apiKey": self.api_key,
            "format": "json",
            "content": str(mdef.get("content") or "json"),
            "orgId": org_id,
            "tblId": stat_id,
            "objL1": obj_l1,
            "itmId": itm_id,
            "prdSe": prd_se,
        }
        if mdef.get("objL2"):
            req["objL2"] = str(mdef["objL2"]).strip()
        if sp:
            req["startPrdDe"] = sp
        if ep:
            req["endPrdDe"] = ep
        if ncnt:
            req["newEstPrdCnt"] = ncnt
        if not sp and not ep and not ncnt:
            req["newEstPrdCnt"] = "1"
        pint = mdef.get("prdInterval")
        if pint is not None and str(pint).strip() != "":
            req["prdInterval"] = str(pint).strip()
        of = mdef.get("outputFields")
        if of is not None and str(of).strip() != "":
            req["outputFields"] = str(of).strip()

        public_req = {k: v for k, v in req.items() if k != "apiKey"}
        err_body: dict[str, Any] = {}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(
                    f"{base}/Param/statisticsParameterData.do",
                    params=req,
                    headers=_JSON_HEADERS,
                )
                r.raise_for_status()
                text = (r.text or "").strip()
                if not text or text.lstrip().startswith("<"):
                    err_body = {
                        "error": "empty_or_html",
                        "request": public_req,
                    }
                    return StatData(
                        stat_id=stat_id,
                        stat_name=tnm,
                        values={},
                        raw_response=err_body,
                    )
                j = _kosis_text_to_json(text)
                if j is None:
                    return StatData(
                        stat_id=stat_id,
                        stat_name=tnm,
                        values={},
                        raw_response={"error": "json_parse", "request": public_req},
                    )
                if (
                    isinstance(j, dict)
                    and j.get("err") is not None
                    and "row" not in j
                ):
                    return StatData(
                        stat_id=stat_id,
                        stat_name=tnm,
                        values={},
                        raw_response={
                            "err": j.get("err"),
                            "errMsg": j.get("errMsg"),
                            "request": public_req,
                        },
                    )
                drows = _rows_from_kosis_body(j)
                if not drows:
                    return StatData(
                        stat_id=stat_id,
                        stat_name=tnm,
                        values={},
                        raw_response={**(j if isinstance(j, dict) else {}), "request": public_req},
                    )
                cell0 = drows[0]
                tnm2 = (cell0.get("TBL_NM") or tnm) or stat_id
                raw_out: dict[str, Any] = {**j, "request": public_req} if isinstance(
                    j, dict) else {
                    "row": drows,
                    "request": public_req,
                    "data": j,
                }
                dt_s = (cell0.get("DT") or "")
                if isinstance(dt_s, (int, float)):
                    dt_s = str(dt_s)
                dt_s = str(dt_s).strip()
                val: float | None = None
                if dt_s:
                    try:
                        val = float(dt_s.replace(",", ""))
                    except ValueError:
                        val = None
                # 반환값 구성: ratio=value(float) / DT=시점 / PRD_DE=주기 / ITM_NM=항목명 / ITM_ID=항목 ID / UNIT_NM=단위명
                vmap: dict[str, Any] = {
                    "value": val,
                    "DT": cell0.get("DT"),
                    "PRD_DE": cell0.get("PRD_DE"),
                    "ITM_NM": cell0.get("ITM_NM"),
                    "ITM_ID": cell0.get("ITM_ID"),
                    "UNIT_NM": cell0.get("UNIT_NM"),
                }
                if val is not None:
                    vmap["ratio"] = val
                vmap = {k: v for k, v in vmap.items() if v is not None}
                u = _kosis_cell_str(cell0.get("UNIT_NM"))
                tp = _kosis_cell_str(cell0.get("PRD_DE"))
                if not tp and cq and (cq.time_period or "").strip():
                    tp = _kosis_cell_str(cq.time_period)
                return StatData(
                    stat_id=stat_id,
                    stat_name=str(tnm2).strip() or stat_id,
                    values=vmap,
                    raw_response=raw_out,
                    official_value=val,
                    unit=u,
                    time_period=tp,
                )
        except httpx.HTTPError as e:
            logger.error("KOSIS Param fetch HTTP: %s", e)
            return StatData(
                stat_id=stat_id,
                stat_name=tnm,
                values={},
                raw_response={"error": "http", "request": public_req, "detail": str(e)[:200]},
            )
        except Exception as e:  # noqa: BLE001
            logger.error("KOSIS Param fetch: %s", e)
            return StatData(
                stat_id=stat_id,
                stat_name=tnm,
                values={},
                raw_response={"error": "exception", "request": public_req, "detail": str(e)[:200]},
            )

    def to_graph_nodes(self, data: StatData) -> list[GraphNode]:
        """KOSIS 결과 → Evidence 그래프 노드 변환"""
        return [GraphNode(node_id=f"evidence:{data.stat_id}",
                          node_type=GraphNodeType.EVIDENCE,
                          label=data.stat_name, properties=data.values)]

    def tag_provenance(self, data: StatData, query: ConnectorQuery) -> ProvenanceRecord:
        """출처 이력 기록"""
        return ProvenanceRecord(source_connector="KOSIS", source_id=data.stat_id,
                                query_used=query.keyword, raw_snapshot=data.raw_response)
