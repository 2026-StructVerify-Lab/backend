"""
storage/db_manager.py — PostgreSQL OLTP 매니저

SIR Tree, Claims, Verification Results, Artifacts를 JSONB로 저장한다.

[박재윤]
- SQLAlchemy async engine 초기화 및 CRUD 구현 담당
- init_db.sql의 테이블 구조에 맞춰 INSERT/SELECT 구현
- 배치 INSERT 최적화 (executemany 활용)
"""
from __future__ import annotations
from structverify.core.schemas import SIRDocument, Claim, VerificationResult
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class DBManager:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

        # TODO [박재윤]: SQLAlchemy async engine 초기화
        #   from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        #   from sqlalchemy.orm import sessionmaker
        #   self.engine = create_async_engine(
        #       self.config.get("url", "postgresql+asyncpg://user:pass@localhost:5432/structverify"),
        #       echo=False,
        #   )
        #   self.AsyncSession = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def save_document(self, doc: SIRDocument) -> None:
        """
        SIR 문서를 PostgreSQL에 저장한다.

        TODO [박재윤]: documents 테이블 INSERT 구현
          INSERT INTO documents (doc_id, source_type, source_uri, sir_json, extracted_at)
          VALUES ($1, $2, $3, $4::jsonb, $5)
          ON CONFLICT (doc_id) DO NOTHING

          - doc_id: str(doc.doc_id)
          - source_type: doc.source_type.value
          - sir_json: doc.model_dump_json() 또는 json.dumps
          - 비동기 세션 사용 (async with self.AsyncSession() as session)
        """
        logger.warning(f"DB 저장 stub: doc {doc.doc_id}")

    async def save_claims(self, claims: list[Claim]) -> None:
        """
        TODO [박재윤]: claims 테이블 배치 INSERT 구현
          INSERT INTO claims (claim_id, doc_id, block_id, sent_id, claim_text,
                              claim_type, check_worthy_score, schema_json)
          VALUES ...
          - executemany 또는 asyncpg copy_records_to_table 활용
          - schema 필드는 JSONB로 저장
        """
        logger.warning(f"DB 저장 stub: {len(claims)} claims")

    async def save_results(self, results: list[VerificationResult]) -> None:
        """
        TODO [박재윤]: verification_results 테이블 배치 INSERT 구현
          INSERT INTO verification_results
            (result_id, claim_id, verdict, confidence, mismatch_type,
             evidence_json, explanation, created_at)
          VALUES ...
          - verdict: result.verdict.value
          - evidence_json: result.evidence.model_dump() if result.evidence else None
        """
        logger.warning(f"DB 저장 stub: {len(results)} results")

    async def save_feedback(self, event) -> None:
        """
        TODO [박재윤]: feedback_events 테이블 INSERT 구현
          INSERT INTO feedback_events
            (event_id, claim_id, feedback_type, original_verdict,
             corrected_verdict, reviewer_note, created_at)
          VALUES ...
        """
        logger.warning(f"DB 저장 stub: feedback {event}")
