"""
core/pipeline.py — v2.1 검증 파이프라인 (13단계)

  ① 입력 → ② 전처리+SIR Tree → ③ 도메인 판별
  → ④ Sentence Candidate Detection + Claim Detection
  → ⑤ Schema Induction → ⑥ Graph Construction
  → ⑦ Retrieval+Evidence Subgraph → ⑧ Verification
  → ⑨ Explanation+Provenance → ⑩ Human Review
  → ⑪ Feedback Logging → ⑫ Adaptation Trigger → ⑬ Report

[김예슬]
- 기존 Step 4의 "수치 포함 문장 필터링 + Claim Detection" 구조를
  "문장 단위 candidate scoring + Claim Detection" 구조로 확장
- has_numeric regex는 보조 surface signal로만 사용
- 실제 검증 후보 여부는 candidate_score / candidate_label 기준으로 판단

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
  Thought→Action→Observation 순환 기반 오케스트레이션 패턴
"""
from __future__ import annotations

from structverify.core.schemas import (
    SourceType,
    SIRDocument,
    VerificationReport,
)
from structverify.preprocessing.extractor import extract_text
from structverify.preprocessing.sir_builder import build_sir
from structverify.agent.runtime_agent import RuntimeAgent
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class VerificationPipeline:
    """
    13단계 검증 파이프라인 오케스트레이터.

    사용법:
        pipeline = VerificationPipeline()
        report = await pipeline.run("https://news.example.com/article", "url")

    Step 4 변경사항:
        기존:
            has_numeric=True 문장 선별 → LLM check-worthiness
        변경:
            sentence candidate scoring → 상위 후보에 대해 LLM check-worthiness

    즉, Claim Detection 앞단에 "Sentence Candidate Detection"이 추가되었다.
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.runtime_agent = RuntimeAgent(config=self.config)

        # TODO: BuilderAgent 초기화 (비동기 백그라운드)
        # TODO: RawStorage, DBManager, DWHManager, GraphStore 초기화

    async def run(self, source: str, source_type: str = "text") -> VerificationReport:
        """
        source 입력을 받아 전체 검증 파이프라인을 수행한다.

        Args:
            source: 원본 입력 (텍스트/URL/PDF 경로 등)
            source_type: "text" | "url" | "pdf" | "docx"

        Returns:
            VerificationReport: 전체 검증 결과 보고서
        """
        src = SourceType(source_type)
        logger.info(f"파이프라인 시작: {src.value}")

        # ─────────────────────────────────────────────
        # Step 1~2: 입력 → 전처리 → SIR Tree
        # ─────────────────────────────────────────────
        # [TODO] 이수민 : 전처리 담당이 타는 구간
        # 원본 입력 → Raw Text 추출 → SIR Tree 변환
        # extract_text와 build_sir 함수 확인
        raw_text = await extract_text(source, src)
        sir_doc = build_sir(
            raw_text,
            src,
            source_uri=source if src == SourceType.URL else None,
        )
        logger.info(f"SIR Tree: {len(sir_doc.blocks)} blocks")

        # TODO: Step 1.5 — 원본 → Raw Storage(S3) 저장
        # TODO: Step 2.5 — SIR Tree → PostgreSQL 적재

        # ─────────────────────────────────────────────
        # Step 3~9: Runtime Agent 실행
        # ─────────────────────────────────────────────
        # Step 3: Domain Classification
        # Step 4: Sentence Candidate Detection + Claim Detection
        #         - 문장 단위 candidate_score / candidate_label 계산
        #         - 상위 후보 문장만 check-worthiness 판별
        # Step 5: Schema Induction
        # Step 6: Graph Construction
        # Step 7: Retrieval + Evidence Subgraph Extraction
        # Step 8: Verification (deterministic)
        # Step 9: Explanation + Provenance
        claims, results, nodes, edges = await self.runtime_agent.process(sir_doc)

        # TODO: Step 10 — Human Review 인터페이스 연동
        # TODO: Step 11 — Feedback Logging (BuilderAgent.log_feedback)
        # TODO: Step 12 — Adaptation Trigger (BuilderAgent._trigger_adaptation)
        # TODO: Step 13 — Report 렌더링 (PDF/HTML)
        # TODO: DWH 적재 (Snowflake/BigQuery)

        report = VerificationReport(
            document=sir_doc,
            claims=claims,
            results=results,
            graph_nodes=nodes,
            graph_edges=edges,
            domain_pack_used=sir_doc.detected_domain,
        )
        logger.info(f"파이프라인 완료: {len(results)} results")
        return report


async def verify_text(text: str, config: dict | None = None) -> VerificationReport:
    """
    최상위 API — 텍스트 입력 → 검증 보고서

    Args:
        text: 검증할 원문 텍스트
        config: 선택적 설정 dict

    Returns:
        VerificationReport
    """
    return await VerificationPipeline(config).run(text, "text")