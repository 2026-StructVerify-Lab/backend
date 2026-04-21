"""
core/pipeline.py — v2.0 검증 파이프라인 (13단계)

  ① 입력 → ② 전처리+SIR Tree → ③ 도메인 판별 → ④ Claim Detection
  → ⑤ Schema Induction → ⑥ Graph Construction → ⑦ Retrieval+Evidence Subgraph
  → ⑧ Verification → ⑨ Explanation+Provenance → ⑩ Human Review
  → ⑪ Feedback Logging → ⑫ Adaptation Trigger → ⑬ Report

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
  Thought→Action→Observation 순환 기반 오케스트레이션 패턴
"""
from __future__ import annotations
from structverify.core.schemas import (
    SourceType, SIRDocument, VerificationReport)
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
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.runtime_agent = RuntimeAgent(config=self.config)
        # TODO: BuilderAgent 초기화 (비동기 백그라운드)
        # TODO: RawStorage, DBManager, DWHManager, GraphStore 초기화

    async def run(self, source: str, source_type: str = "text") -> VerificationReport:
        src = SourceType(source_type)
        logger.info(f"파이프라인 시작: {src.value}")

        # ── Step 1~2: 입력 → 전처리 → SIR Tree ──
        # [TODO] 이수민 : 전처리 담당이 타는 구간 (원본 → Raw Text → SIR Tree)입니다. extract_text와 build_sir 함수 확인 
        raw_text = await extract_text(source, src)
        sir_doc = build_sir(raw_text, src,
                            source_uri=source if src == SourceType.URL else None)
        logger.info(f"SIR Tree: {len(sir_doc.blocks)} blocks")

        # TODO: Step 1.5 — 원본 → Raw Storage(S3) 저장
        # TODO: Step 2.5 — SIR Tree → PostgreSQL 적재

        # ── Step 3~9: Runtime Agent 실행 ──
        claims, results, nodes, edges = await self.runtime_agent.process(sir_doc)

        # TODO: Step 10 — Human Review 인터페이스 연동
        # TODO: Step 11 — Feedback Logging (BuilderAgent.log_feedback)
        # TODO: Step 12 — Adaptation Trigger (BuilderAgent._trigger_adaptation)
        # TODO: Step 13 — Report 렌더링 (PDF/HTML)
        # TODO: DWH 적재 (Snowflake/BigQuery)

        report = VerificationReport(
            document=sir_doc, claims=claims, results=results,
            graph_nodes=nodes, graph_edges=edges,
            domain_pack_used=sir_doc.detected_domain,
        )
        logger.info(f"파이프라인 완료: {len(results)} results")
        return report


async def verify_text(text: str, config: dict | None = None) -> VerificationReport:
    """최상위 API — 텍스트 입력 → 검증 보고서"""
    return await VerificationPipeline(config).run(text, "text")
