"""
storage/db_manager.py — PostgreSQL OLTP 매니저

SIR Tree, Claims, Verification Results, Artifacts를 JSONB로 저장한다.
"""
from __future__ import annotations
from structverify.core.schemas import SIRDocument, Claim, VerificationResult
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class DBManager:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # TODO: SQLAlchemy async engine 초기화

    async def save_document(self, doc: SIRDocument) -> None:
        """
        SIR 문서를 PostgreSQL에 저장한다.
        TODO: INSERT INTO documents (doc_id, source_type, sir_json, ...) 구현
        """
        logger.warning(f"DB 저장 stub: doc {doc.doc_id}")

    async def save_claims(self, claims: list[Claim]) -> None:
        """TODO: INSERT INTO claims 배치 구현"""
        logger.warning(f"DB 저장 stub: {len(claims)} claims")

    async def save_results(self, results: list[VerificationResult]) -> None:
        """TODO: INSERT INTO verification_results 배치 구현"""
        logger.warning(f"DB 저장 stub: {len(results)} results")
