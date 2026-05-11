"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-26
# 수정 내용: __init__ psycopg2 연결 구현, save_claims INSERT 구현

# [DONE] __init__ DB 연결 초기화
# [DONE] save_claims 배치 INSERT 구현
# [DONE] save_results 배치 INSERT 구현 (claimed_value, true_value, deviation 계산 포함)
# [DONE] save_claims ON CONFLICT (claim_id) DO NOTHING 추가 (중복 실행 방지)
# [DONE] save_claims/save_results 매번 새 연결로 변경 (long-running 연결 끊김 방지)
# [DONE] save_claims domain 파라미터 추가
# [TODO] save_document 구현
# [TODO] save_feedback 구현
"""
from __future__ import annotations
import os
from structverify.core.schemas import SIRDocument, Claim, VerificationResult
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class DBManager:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        from dotenv import load_dotenv
        load_dotenv()

    def _get_conn(self):
        import psycopg2
        return psycopg2.connect(
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
        """
        logger.warning(f"DB 저장 stub: doc {doc.doc_id}")

    async def save_claims(self, claims: list[Claim], domain: str = None) -> None:
        # [v2] - 박재윤: 매번 새 연결 (long-running 연결 끊김 방지)
        conn = self._get_conn()
        cur = conn.cursor()
        for claim in claims:
            cur.execute(
                "INSERT INTO requests (request_id, domain, submitted_at) VALUES (%s, %s, NOW()) ON CONFLICT (request_id) DO NOTHING",
                (str(claim.doc_id), domain)
            )
            indicator = claim.schema.indicator if claim.schema else None
            value     = claim.schema.value     if claim.schema else None
            unit      = claim.schema.unit      if claim.schema else None
            time_ref  = claim.schema.time_period if claim.schema else None
            cur.execute("""
                INSERT INTO claims (claim_id, request_id, field_name, field_value,
                                  unit, is_approximate, modifier, parent_path,
                                  time_reference, context)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (claim_id) DO NOTHING
            """, (
                str(claim.claim_id), str(claim.doc_id),
                indicator, value, unit,
                False, None, None,
                time_ref, claim.claim_text
            ))
        conn.commit()
        cur.close()
        conn.close()

    async def save_results(self, results: list[VerificationResult], claims: list[Claim] = None) -> None:
        # [v2] - 박재윤: 매번 새 연결 (long-running 연결 끊김 방지)
        conn = self._get_conn()
        cur = conn.cursor()

        claim_value_map = {}
        if claims:
            for c in claims:
                if c.schema and c.schema.value is not None:
                    claim_value_map[str(c.claim_id)] = c.schema.value

        for result in results:
            ev = result.evidence
            claimed_value = claim_value_map.get(str(result.claim_id))
            true_value = ev.official_value if ev else None

            deviation = None
            if claimed_value and true_value and true_value != 0:
                deviation = abs(claimed_value - true_value) / abs(true_value)

            cur.execute("""
                INSERT INTO results (result_id, claim_id, truth_id,
                                    claimed_value, true_value, deviation,
                                    match_status, reason, explanation, judged_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (result_id) DO NOTHING
            """, (
                str(result.result_id),
                str(result.claim_id),
                None,
                claimed_value,
                true_value,
                deviation,
                result.verdict.value,
                result.mismatch_type.value if result.mismatch_type else None,
                result.explanation,
                result.created_at,
            ))

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Results 저장 완료: {len(results)}건")

    async def save_feedback(self, event) -> None:
        """
        TODO [박재윤]: feedback_events 테이블 INSERT 구현
          INSERT INTO feedback_events
            (event_id, claim_id, feedback_type, original_verdict,
             corrected_verdict, reviewer_note, created_at)
          VALUES ...
        """
        logger.warning(f"DB 저장 stub: feedback {event}")