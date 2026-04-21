"""
storage/dwh_manager.py — DWH 매니저 (Snowflake / BigQuery / ClickHouse)

검증 결과 로그, 모델 성능, 비용 데이터를 분석 계층에 적재한다.
v2: Snowflake 커넥터 추가.
"""
from __future__ import annotations
import os
from typing import Any
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class DWHManager:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "snowflake")

    def _get_snowflake_conn(self):
        """
        Snowflake 연결 생성
        TODO: snowflake.connector.connect() 호출 구현
        """
        cfg = self.config.get("snowflake", {})
        # import snowflake.connector
        # return snowflake.connector.connect(
        #     account=os.environ.get(cfg.get("account_env", "")),
        #     user=os.environ.get(cfg.get("user_env", "")),
        #     password=os.environ.get(cfg.get("password_env", "")),
        #     warehouse=cfg.get("warehouse", "STRUCTVERIFY_WH"),
        #     database=cfg.get("database", "STRUCTVERIFY_DB"),
        #     schema=cfg.get("schema", "PUBLIC"),
        # )
        logger.warning("Snowflake 연결 stub")
        return None

    async def load_verification_logs(self, records: list[dict[str, Any]]) -> None:
        """
        검증 결과 로그를 DWH에 적재한다.

        Snowflake:
          INSERT INTO verification_logs (result_id, claim_id, verdict, confidence, ...)
          VALUES (%s, %s, %s, %s, ...)

        TODO: snowflake cursor.executemany() 배치 INSERT 구현
        TODO: BigQuery/ClickHouse 분기 구현
        """
        logger.warning(f"DWH 적재 stub ({self.provider}): {len(records)} records")

    async def load_model_metrics(self, metrics: dict[str, Any]) -> None:
        """
        모델 성능 지표를 DWH에 적재한다.
        TODO: INSERT INTO model_metrics 구현
        """
        logger.warning(f"DWH 모델 지표 stub: {metrics}")

    async def load_llm_costs(self, cost_records: list[dict[str, Any]]) -> None:
        """
        LLM 호출 비용 데이터를 DWH에 적재한다.
        TODO: INSERT INTO llm_cost_logs 구현
        """
        logger.warning(f"DWH 비용 stub: {len(cost_records)} records")
