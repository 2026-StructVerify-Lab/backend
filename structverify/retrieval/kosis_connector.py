"""
retrieval/kosis_connector.py — KOSIS Open API 커넥터 (과제 핵심)

[참고] KOSIS Open API — https://kosis.kr/openapi/index/index.jsp
  getList(통계표 목록 검색), getStatData(상세 데이터 조회) 엔드포인트 사용
"""
from __future__ import annotations
import os
from typing import Any
from datetime import datetime
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
        """
        KOSIS 통계표 목록 검색 (getList)
        TODO: httpx GET 호출 구현
        # params: apiKey, format=json, vwCd=MT_ZTITLE, searchNm=query.keyword
        """
        logger.warning(f"KOSIS search stub: {query.keyword}")
        return [StatRecord(stat_id="DT_1EA1019", stat_name="경영주 연령별 농가",
                           org_name="통계청", available_periods=["2024"],
                           relevance_score=0.95)]

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
