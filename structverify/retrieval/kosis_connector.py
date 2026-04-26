"""
retrieval/kosis_connector.py — KOSIS Open API 커넥터 (과제 핵심)

[참고] KOSIS Open API — https://kosis.kr/openapi/index/index.jsp
  getList(통계표 목록 검색), getStatData(상세 데이터 조회) 엔드포인트 사용

- 담당자: 신준수
"""
# 수정자: 신준수
# 수정 날짜: 2026-04-22
# 수정 내용: KOSIS API search() 실제 호출 구현

# [DONE] KOSIS API search() 구현
# [TODO] KOSIS API fetch() 구현
# [TODO] 응답 파싱 로직 정교화

from __future__ import annotations
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
        # 한 번에 몇 개까지 가져올지: 나중에 `search_and_fetch`가 relevance 높은 걸 골라도, 후보 풀은 이 개수
        n_max = int(self.config.get("search_result_count", 20))  # resultCount(한 페이지 상한)
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
            # 기간 필드(있으면): 후보에 "어느 시점 자료인지" 힌트
            periods: list[str] = []
            for k in ("STRT_PRD_DE", "END_PRD_DE"):
                v = item.get(k)  # 수록기간 시작/끝(가이드 필드명)
                if v is not None and str(v).strip() and str(v).strip() not in periods:
                    periods.append(str(v).strip())
            # relevance_score: KOSIS가 RANK로 이미 정렬이므로, 앞 index일수록 점수 높게(1.0 → 내려감)
            # `BaseConnector.search_and_fetch` 는 max(relevance_score)로 하나 고름
            rel = 1.0 if n <= 1 else max(0.05, 1.0 - (i / (n - 1)) * 0.95)  # 순위→상대 점수 -> kosis 응답이 rank 순으로 정렬되어 있어 임의로 정렬해봄
            records.append(StatRecord(
                stat_id=tid, stat_name=tnm, org_name=org, available_periods=periods,
                relevance_score=rel,
            ))
        logger.info(f"KOSIS API 검색 완료: {len(records)}개 (응답 {n}행)")
        return records

    async def fetch(self, stat_id: str, params: dict[str, Any]) -> StatData:
        """
        KOSIS 상세 데이터 조회 (getStatData)
        TODO: httpx GET 호출 구현
        # params: apiKey, method=getStatData, orgId=101, tblId=stat_id
        """
        logger.warning(f"KOSIS fetch stub: {stat_id}")
        return StatData(stat_id=stat_id, stat_name="경영주 연령별 농가",
                        values={"total": 166558, "age_65_plus": 106877, "ratio": 64.2})

    def to_graph_nodes(self, data: StatData) -> list[GraphNode]:
        """KOSIS 결과 → Evidence 그래프 노드 변환"""
        return [GraphNode(node_id=f"evidence:{data.stat_id}",
                          node_type=GraphNodeType.EVIDENCE,
                          label=data.stat_name, properties=data.values)]

    def tag_provenance(self, data: StatData, query: ConnectorQuery) -> ProvenanceRecord:
        """출처 이력 기록"""
        return ProvenanceRecord(source_connector="KOSIS", source_id=data.stat_id,
                                query_used=query.keyword, raw_snapshot=data.raw_response)
