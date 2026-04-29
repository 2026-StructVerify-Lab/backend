"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-29
# 수정 내용: DBManager 초기화 및 save_claims, save_results 연동 구현

# [DONE] DBManager 초기화 연동
# [DONE] save_claims 파이프라인 연결
# [DONE] save_results 파이프라인 연결
# [TODO] RawStorage 초기화 (MinIO/S3 업로드)
# [TODO] DWHManager 초기화 (Snowflake)
# [TODO] GraphStore 초기화 (Neo4j)
# [TODO] save_document 구현 (db_manager.py)

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
- 전체 파이프라인 흐름 및 Agent 간 연결 관리

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
from structverify.storage.db_manager import DBManager
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

        # [v1] - 박재윤: DBManager 초기화 연동
        self.db_manager = DBManager(config=self.config.get("database", {}))

        # TODO [김예슬]: BuilderAgent 초기화 (비동기 백그라운드 학습 루프)
        # self.builder_agent = BuilderAgent(config=self.config)

        # TODO [박재윤]: RawStorage 초기화 (MinIO/S3 업로드)
        # self.raw_storage = RawStorage(config=self.config.get("storage", {}))

        # TODO [박재윤]: DWHManager 초기화 (Snowflake)
        # self.dwh_manager = DWHManager(config=self.config.get("dwh", {}))

        # TODO [박재윤]: GraphStore 초기화 (Neo4j)
        # self.graph_store = GraphStore(config=self.config.get("graph", {}))

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
        # TODO [이수민]: extractor.py — URL/PDF/DOCX 실제 추출 로직 구현
        #   - URL: trafilatura.extract() 호출
        #   - PDF: PyMuPDF(fitz) 페이지별 텍스트 추출
        #   - DOCX: python-docx 단락 추출
        raw_text = await extract_text(source, src)

        # TODO [이수민]: sir_builder.py — SIR Tree 변환 검증
        #   - block 분할 + 문장 분리 + 절대 offset 보정 확인
        #   - entity_refs, event_refs 추출 (현재 regex placeholder → NER 교체 예정)
        sir_doc = build_sir(
            raw_text,
            src,
            source_uri=source if src == SourceType.URL else None,
        )
        logger.info(f"SIR Tree: {len(sir_doc.blocks)} blocks")

        # TODO [박재윤]: Step 1.5 — 원본 텍스트 → Raw Storage(S3/MinIO) 저장
        # await self.raw_storage.upload(source, raw_text, metadata={"source_type": src.value})

        # TODO [박재윤]: Step 2.5 — SIR Document → PostgreSQL 저장
        # await self.db_manager.save_document(sir_doc)

        # Step 3~9: Runtime Agent 실행
        claims, results, nodes, edges = await self.runtime_agent.process(sir_doc)

        # [v1] - 박재윤: Claims → PostgreSQL 저장
        if claims:
            await self.db_manager.save_claims(claims)
            logger.info(f"Claims 저장 완료: {len(claims)}건")

        # [v1] - 박재윤: Results → PostgreSQL 저장
        if results:
            await self.db_manager.save_results(results)
            logger.info(f"Results 저장 완료: {len(results)}건")

        # TODO [박재윤]: Nodes/Edges → Neo4j 저장
        # await self.graph_store.merge_nodes(nodes)
        # await self.graph_store.merge_edges(edges)

        # TODO [김예슬]: Step 10 — Human Review 인터페이스 연동
        # TODO [김예슬]: Step 11 — Feedback Logging
        # TODO [김예슬]: Step 12 — Adaptation Trigger
        # TODO [김예슬]: Step 13 — Report 렌더링

        # TODO [박재윤]: DWH 적재 (검증 로그, 모델 성능, LLM 비용)
        # await self.dwh_manager.load_verification_logs([...])

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
