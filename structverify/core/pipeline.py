"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-29
# 수정 내용: DBManager 초기화 및 save_claims, save_results 연동 구현

# [DONE] DBManager 초기화 연동
# [DONE] save_claims 파이프라인 연결
# [DONE] save_results 파이프라인 연결
# [DONE] 기사 텍스트 해시로 doc_id 고정 (재실행 시 중복 방지)
# [DONE] save_claims source_type 파라미터 추가
# [TODO] RawStorage 초기화 (MinIO/S3 업로드)
# [TODO] DWHManager 초기화 (Snowflake)
# [TODO] GraphStore 초기화 (Neo4j)
# [TODO] save_document 구현 (db_manager.py)
# [김예슬 - 2026-04-30 / v3]
# - config 기본값 load_config() 적용
# - feedback/학습: enable_feedback=false 이면 Step 11~12 skip
# - runtime_agent에 kosis llm config 전달 (LLM Agent catalog 검색용)

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
from structverify.core.config_loader import load_config
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
        # [v3 김예슬] config 없으면 default.yaml 자동 로드
        self.config = config if config is not None else load_config()

        # runtime_agent에 kosis + llm 설정 합쳐서 전달
        # → KOSISConnector가 LLM Agent(catalog 검색)에 llm 설정 필요
        agent_config = {
            **self.config,
            "kosis": {
                **self.config.get("kosis", {}),
                "llm": self.config.get("llm", {}),
            },
        }
        self.runtime_agent = RuntimeAgent(config=agent_config)

        # [v1] - 박재윤: DBManager 초기화 연동
        self.db_manager = DBManager(config=self.config.get("database", {}))

        # [v3 김예슬] 피드백/학습 활성화 여부 (false면 Step 11~12 skip)
        self.enable_feedback = bool(self.config.get("enable_feedback", False))

        # TODO [김예슬]: BuilderAgent 초기화 (비동기 백그라운드 학습 루프)
        # self.builder_agent = BuilderAgent(config=self.config)

        # TODO [박재윤]: RawStorage 초기화 (MinIO/S3 업로드)
        # self.raw_storage = RawStorage(config=self.config.get("storage", {}))

        # TODO [박재윤]: DWHManager 초기화 (Snowflake)
        # self.dwh_manager = DWHManager(config=self.config.get("dwh", {}))

        # TODO [박재윤]: GraphStore 초기화 (Neo4j)
        # self.graph_store = GraphStore(config=self.config.get("graph", {}))

    async def run(
        self,
        source: str,
        source_type: str = "text",
        source_text: str | None = None,
    ) -> VerificationReport:
        """
        source 입력을 받아 전체 검증 파이프라인을 수행한다.

        Args:
            source: 원본 입력 (텍스트/URL/PDF 경로 등)
            source_type: "text" | "url" | "pdf" | "docx"
            source_text: *옵션* — 이미 추출된 본문 텍스트가 있으면 전달.
                URL/PDF 입력에서 sv_platform이 미리 추출해 Job.source_data에
                저장한 뒤 그 텍스트를 그대로 넘겨주면 *중복 추출 없이* 파이프라인
                진행. (실시간 partial 응답 매칭에도 사용됨.)
                None이면 source에서 extract_text() 직접 호출.

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
        if source_text is not None and source_text.strip():
            # sv_platform이 사전 추출해 넘긴 경우 — 중복 호출 회피
            raw_text = source_text
            logger.info(f"파이프라인: 사전 추출 본문 사용 ({len(raw_text)}자)")
        else:
            raw_text = await extract_text(source, src)

        # TODO [이수민]: sir_builder.py — SIR Tree 변환 검증
        #   - block 분할 + 문장 분리 + 절대 offset 보정 확인
        #   - entity_refs, event_refs 추출 (현재 regex placeholder → NER 교체 예정)
        sir_doc = build_sir(
            raw_text,
            src,
            source_uri=source if src == SourceType.URL else None,
        )
        # [v2] - 박재윤: 기사 텍스트 해시로 doc_id 고정 (재실행 시 중복 방지)
        import hashlib
        from uuid import UUID
        text_hash = hashlib.md5(raw_text.encode()).hexdigest()
        sir_doc.doc_id = UUID(text_hash)
        # [2026-05-21] URL/PDF 입력의 추출된 본문을 SIRDocument에 보존.
        # sv_platform이 Job.source_data로 복사해 프론트 "원문" 패널 렌더에 사용.
        sir_doc.raw_text = raw_text
        logger.info(f"SIR Tree: {len(sir_doc.blocks)} blocks")

        # TODO [박재윤]: Step 1.5 — 원본 텍스트 → Raw Storage(S3/MinIO) 저장
        # await self.raw_storage.upload(source, raw_text, metadata={"source_type": src.value})

        # TODO [박재윤]: Step 2.5 — SIR Document → PostgreSQL 저장
        # await self.db_manager.save_document(sir_doc)

        # Step 3~9: Runtime Agent 실행
        claims, results, nodes, edges = await self.runtime_agent.process(sir_doc)

        # [v1] - 박재윤: Claims → PostgreSQL 저장
        if claims:
            await self.db_manager.save_claims(claims, domain=sir_doc.detected_domain, source_type=src.value)
            logger.info(f"Claims 저장 완료: {len(claims)}건")

        # [v1] - 박재윤: Results → PostgreSQL 저장
        if results:
            await self.db_manager.save_results(results, claims)
            logger.info(f"Results 저장 완료: {len(results)}건")

        # TODO [박재윤]: Nodes/Edges → Neo4j 저장
        # await self.graph_store.merge_nodes(nodes)
        # await self.graph_store.merge_edges(edges)

        # Step 10~13: 피드백/학습 루프 (enable_feedback=true 일 때만 활성화)
        # [v3 김예슬] enable_feedback=false (기본값) → skip
        # 활성화: config.yaml에서 enable_feedback: true
        if self.enable_feedback:
            pass
            # TODO [김예슬]: Step 10 — Human Review 인터페이스 연동
            # TODO [김예슬]: Step 11 — Feedback Logging
            # TODO [김예슬]: Step 12 — Adaptation Trigger
            # TODO [김예슬]: Step 13 — Report 렌더링
        else:
            logger.debug("Step 10~13 skip (enable_feedback=false)")

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


# [v2 - 김예슬] 문서 입력(URL/PDF/DOCX/TEXT) 공개 진입점
async def verify_document(
    source: str,
    source_type: str = "url",
    config: dict | None = None,
    source_text: str | None = None,
) -> VerificationReport:
    """
    최상위 API — 문서 입력(URL/PDF/DOCX/TEXT) → 검증 보고서.

    Args:
        source: URL · 파일 경로 · 또는 본문 텍스트
        source_type: "url" | "pdf" | "docx" | "text"
        config: 선택적 설정 dict
        source_text: 이미 추출된 본문이 있으면 전달(중복 추출 방지)

    Returns:
        VerificationReport
    """
    return await VerificationPipeline(config).run(source, source_type, source_text)


# [v2 - 김예슬] 객체형 진입점 — 같은 config로 여러 번 검증할 때 (함수형은 일회성).
class VerificationEngine:
    """config를 한 번 주입해 두고 여러 문서를 검증하는 객체형 진입점.

    - 함수형(`verify_text`/`verify_document`): 매번 config를 받는 *일회성* 호출에 편함.
    - 객체형(`VerificationEngine`): 엔진을 만들어 두고 재사용.

        engine = VerificationEngine(config)
        r1 = await engine.verify_text("문장1")
        r2 = await engine.verify_document(url, source_type="url")
    """

    def __init__(self, config: dict | None = None):
        self._pipeline = VerificationPipeline(config)

    async def verify_text(self, text: str) -> VerificationReport:
        return await self._pipeline.run(text, "text")

    async def verify_document(
        self,
        source: str,
        source_type: str = "url",
        source_text: str | None = None,
    ) -> VerificationReport:
        return await self._pipeline.run(source, source_type, source_text)