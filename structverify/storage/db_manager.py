"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-26
# 수정 내용: __init__ psycopg2 연결 구현, save_claims INSERT 구현

# [DONE] __init__ DB 연결 초기화
# [DONE] save_claims 배치 INSERT 구현
# [TODO] save_document 구현
# [TODO] save_results 구현
# [TODO] save_feedback 구현
"""
from __future__ import annotations
from structverify.core.schemas import SIRDocument, Claim, VerificationResult
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class DBManager:
    def __init__(self, config: dict | None = None):
      self.config = config or {}
       
      import psycopg2
      from dotenv import load_dotenv
      load_dotenv()
      
      self.conn = psycopg2.connect(
          host=os.getenv("POSTGRES_HOST"),
          port=os.getenv("POSTGRES_PORT"),
          dbname=os.getenv("POSTGRES_DB"),
          user=os.getenv("POSTGRES_USER"),
          password=os.getenv("POSTGRES_PASSWORD")
      )

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
      cur = self.conn.cursor()
      
      for claim in claims:
          cur.execute("""
              INSERT INTO claims (claim_id, request_id, field_name, field_value,
                                unit, is_approximate, modifier, parent_path,
                                time_reference, context)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          """, (
              str(claim.claim_id),
              str(claim.doc_id),
              claim.schema.indicator,
              claim.schema.value,
              claim.schema.unit,
              False,
              None,
              None,
              claim.schema.time_period,
              claim.claim_text
          ))
      
      self.conn.commit()
      cur.close()

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
