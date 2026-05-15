"""
agent/runtime_agent.py — Runtime Verification Agent (Agent A)

실시간 검증 요청을 처리하는 메인 Agent. ReAct 패턴 기반.
Thought → Action(Tool Call) → Observation 순환을 통해 파이프라인 제어.

[김예슬 - 2026-04-22]
- Step 3~9 오케스트레이션 전체 담당
- ReAct 패턴으로 각 스텝을 Action으로 정의하고 순차 실행

[김예슬 - 2026-04-23]
- classify_domain() 반환값 튜플 대응: str → (domain, domain_desc)
- config["detected_domain"] → self.config["detected_domain"] 버그 수정
- domain_desc를 self.config에 저장하여 schema_inductor 힌트로 활용

[김예슬 - 2026-04-24]
- induce_schemas Action 설명 업데이트:
  · 기존: HCX-003 generate_json() → JSON 파싱 (실패 가능)
  · 변경: HCX-007 Structured Outputs → JSON Schema 보장 (파싱 실패 없음)
- Action별 사용 모델/API 업데이트:
  · classify_domain   → HCX-DASH-002 (v3 API)
  · score_candidate   → HCX-DASH-002 (v3 API)
  · check_worthiness  → HCX-003 (v1 API)
  · induce_schemas    → HCX-007 Structured Outputs (v3 API)
  · generate_explain  → HCX-003 (v1 API)

[ReAct 패턴 설명]
  LLM이 단순히 답변을 생성하는 것이 아니라, 매 스텝마다 다음을 반복합니다:
    Thought  : "현재 상태에서 무엇을 해야 하는가?" (LLM 내부 추론)
    Action   : 구체적인 도구(함수) 호출
    Observation: 도구 호출 결과를 관찰하고 다음 Thought 수행

  Action → 사용 모델/API 매핑:
    classify_domain   → HCX-DASH-002 (v3, 경량)
    score_candidate   → HCX-DASH-002 (v3, 경량)
    check_worthiness  → HCX-003 (v1, 중량)
    induce_schemas    → HCX-007 Structured Outputs (v3, JSON 보장)
    build_graph       → 내부 로직 (LLM 미사용)
    retrieve_evidence → KOSIS Open API (LLM 미사용)
    verify_claim      → Deterministic 수치 비교 (LLM 미사용)
    generate_explain  → HCX-003 (v1, 중량)

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct


# [박재윤 - 2026-05-12]
# - Step 7~9 asyncio.gather + Semaphore(3)으로 병렬화 (직렬 → 병렬)

# [DONE] Step 7~9 병렬화 (asyncio.gather + Semaphore 3)

# [2026-05-14 | 이수민] memory/v1: DocumentWorkingMemory 통합
#   - 파이프라인 시작 시 memory 생성 (doc 처리 동안만 유효, 종료 시 소멸)
#   - Step 3: memory.record_domain()  — 도메인 + 설명 기록
#   - Step 4.5: memory.record_anchor_year() + record_temporal()
#   - Step 5: memory.record_claims() — claims + metric_to_claims 자동 인덱싱
#   - Step 8: verify_claim()에 memory 전달 → 도메인 가드 활성화
#            성공한 stat_id는 memory.record_stat_id_used()로 캐시
#   - 종료 시 memory.stats() 로깅
"""
from __future__ import annotations

from structverify.core.schemas import (
    Claim, SIRDocument, VerificationResult, GraphNode, GraphEdge,
)
from structverify.detection.domain_classifier import classify_domain
from structverify.detection.claim_detector import detect_claims
from structverify.detection.schema_inductor import induce_schemas
from structverify.graph.graph_builder import build_claim_graph
from structverify.graph.document_graph import build_document_temporal_graph
from structverify.graph.claim_graph import ClaimGraph
from structverify.retrieval.query_builder import build_query
from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
from structverify.retrieval.kosis_connector import KOSISConnector
from structverify.verification.verifier import verify_claim
from structverify.explanation.explainer import generate_explanation
from structverify.memory import DocumentWorkingMemory  # [이수민 2026-05-14]
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class RuntimeAgent:
    """
    Agent A: 실시간 검증 처리.
    Step 3~9를 순차 실행하며 ReAct 패턴으로 파이프라인을 오케스트레이션한다.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # [v3 김예슬] kosis config에 llm 포함 → CatalogSearchTool LLM Agent가 llm 설정 사용
        kosis_cfg = {
            **self.config.get("kosis", {}),
            "llm": self.config.get("llm", {}),
        }
        self.kosis = KOSISConnector(config=kosis_cfg)

        # TODO [김예슬]: Graph Store 초기화 (Neo4j 노드/엣지 실시간 저장용)
        # from structverify.graph.graph_store import GraphStore
        # self.graph_store = GraphStore(config=self.config.get("graph", {}))

    async def process(self, sir_doc: SIRDocument) -> tuple[
        list[Claim], list[VerificationResult], list[GraphNode], list[GraphEdge]
    ]:
        """
        SIR 문서 → 전체 검증 파이프라인 실행 (Step 3~9).

        Returns:
            (claims, results, graph_nodes, graph_edges)
        """
        # ── [이수민 2026-05-14] DocumentWorkingMemory 생성 ────────────
        # 이 doc 처리 동안만 살아있는 in-memory 컨텍스트.
        # 각 step의 결과를 누적하여 verifier 도메인 가드 등에서 활용.
        from uuid import uuid4
        memory = DocumentWorkingMemory(
            doc_id=str(sir_doc.doc_id),
            run_id=str(uuid4())[:8],
            source_uri=sir_doc.source_uri,
        )

        # ── Action: classify_domain ──────────────────────────────────
        # Tool: HCX-DASH-002 (v3 API, 경량)
        # Thought: "이 문서의 도메인이 무엇인가?"
        # Observation: domain 문자열 + 설명
        domain, domain_desc = await classify_domain(sir_doc, self.config)
        self.config["detected_domain"] = domain
        self.config["detected_domain_desc"] = domain_desc
        memory.record_domain(domain, domain_desc)  # [이수민 2026-05-14]
        logger.info(f"[Agent A] Step 3 classify_domain → {domain} ({domain_desc})")

        # ── Action: detect_claims ────────────────────────────────────
        # [4-1] candidate_scorer: HCX-DASH-002 (v3, 경량) → 0~1 점수
        # [4-2] claim_detector:   HCX-003 (v1, 중량) → check-worthiness
        # Thought: "검증 가능한 주장 문장을 찾아야 한다"
        # Observation: Claim 객체 리스트
        # TODO [김예슬]: domain-packs 기반 도메인별 few-shot 예시 주입
        claims = await detect_claims(sir_doc, self.config)
        logger.info(f"[Agent A] Step 4 detect_claims → {len(claims)}건")

        if not claims:
            logger.info("[Agent A] 검증 가능한 주장 없음 — 파이프라인 종료")
            return [], [], [], []

        # ── [v6 멀티홉] Action: build_document_temporal_graph ───────────
        # Tool: HCX-007 Structured Outputs (v3, 1회 호출)
        # Thought: "문서 전체에서 anchor 시점 + 모든 시간 표현 + coref를
        #           한 번에 추출해서 그래프에 박아두자."
        # Observation: DocumentNode + TemporalExprNodes + ResolvedTimeNodes
        #
        # 이 단계가 schema_inductor 앞에 와야 함:
        #   schema_inductor가 그래프 traverse 결과를 prompt hint로 사용
        # 룰 매핑 일체 없음 — LLM이 anchor와 모든 표현을 동시에 풀어냄
        doc_nodes, doc_edges = await build_document_temporal_graph(sir_doc, self.config)
        # [이수민 2026-05-14] anchor_year + 시간 표현 매핑을 working memory에 기록
        for node in doc_nodes:
            if node.node_type.value == "document":
                ay = node.properties.get("anchor_year")
                if ay:
                    memory.record_anchor_year(int(ay))
            elif node.node_type.value == "temporal_expr":
                expr = node.properties.get("expression")
                resolved = node.properties.get("resolved_value")
                if expr and resolved:
                    memory.record_temporal(expr, resolved)
        logger.info(f"[Agent A] Step 4.5 build_document_temporal_graph → "
                    f"{len(doc_nodes)} nodes, {len(doc_edges)} edges")

        # ── [v4 김예슬] Context Window 부착 ────────────────────────────
        # 각 claim에 앞뒤 문장 context를 붙여서 LLM이 맥락을 이해할 수 있게 함
        # 예: "이는 20년 새 2.6배 증가한 것이다" → 앞 문장 "쉬었음 청년이 21만7천명"도 함께 전달
        # schema_inductor + query_builder에서 context_text 활용
        for claim in claims:
            claim.context_text = _get_context_window(claim, sir_doc, window=2)

        # [v6] document graph만으로 ClaimGraph facade 생성
        # 이 시점에서는 claim/metric 노드는 없지만, schema_inductor가
        # claim.graph_anchor_id로 sentence를 찾아 시점 traverse만 하면 됨
        pre_graph = ClaimGraph(doc_nodes, doc_edges)

        # ── Action: induce_schemas ───────────────────────────────────
        # Tool: HCX-007 Structured Outputs (v3 API)
        # [v6] graph 전달 → "작년" 같은 표현이 그래프에서 "2023"으로 resolved
        #     되어 prompt hint로 들어감. LLM은 그걸 그대로 time_period에 사용.
        claims = await induce_schemas(claims, self.config, graph=pre_graph)
        # [이수민 2026-05-14] claims를 working memory에 기록 (metric_to_claims 자동 인덱싱)
        memory.record_claims(claims)
        logger.info(f"[Agent A] Step 5 induce_schemas → schemas attached "
                    f"(memory: {len(memory.metric_to_claims)} metrics)")

        # ── Action: build_claim_graph ────────────────────────────────
        # [v6] 기존 ClaimNode/MetricNode/EntityNode/COMPARE 생성
        all_nodes, all_edges = build_claim_graph(claims, sir_doc=sir_doc)
        # document temporal 그래프 합치기
        all_nodes.extend(doc_nodes)
        all_edges.extend(doc_edges)
        logger.info(f"[Agent A] Step 6 build_claim_graph → {len(all_nodes)} nodes")

        # [v6] 전체 그래프로 ClaimGraph 재생성 (verifier/explainer가 사용)
        full_graph = ClaimGraph(all_nodes, all_edges)

        # ── 각 주장별 Step 7~9 (병렬 처리) ─────────────────────────
        # [박재윤 - 2026-05-12]: asyncio.gather + Semaphore(3)으로 병렬화
        import asyncio
        sem = asyncio.Semaphore(3)

        async def process_one_claim(claim):
            claim_nid = f"claim:{claim.claim_id.hex[:8]}"
            async with sem:
                # Action: retrieve_evidence (KOSIS API)
                # Tool: KOSIS Open API — pgvector 검색 → LLM 리랭킹 → 실제 수치 조회
                # Thought: "KOSIS에서 공식 수치를 가져와야 한다"
                # Observation: Evidence {official_value, stat_table_id, ...}
                # TODO [신준수]: kosis_connector.py 실제 HTTP 호출 구현
                # TODO [신준수]: query_builder.py ClaimSchema → KOSIS 파라미터 변환
                query = build_query(claim)
                evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
                    self.kosis, query, claim_nid,
                )
                all_nodes.extend(ev_nodes)
                all_edges.extend(ev_edges)
                logger.info(f"[Agent A] Step 7 retrieve_evidence → {str(evidence)[:80] if evidence else None}")

                # Action: verify_claim (Deterministic, LLM 미개입)
                # [v6] graph 전달 → schema.time_period가 "작년"이어도
                #      그래프에서 멀티홉 traverse로 "2023" 찾아서 KOSIS row 매칭
                # [v7 이수민 2026-05-14] memory 전달 → 도메인 가드 활성화
                #      evidence.category_path와 memory.domain 불일치 시 UNVERIFIABLE
                result = verify_claim(claim, evidence, self.config,
                                      graph=full_graph, memory=memory)
                # [이수민 2026-05-14] 성공한 stat_id를 memory에 캐시
                if (result.verdict.value == "match"
                        and evidence and evidence.stat_table_id
                        and claim.schema and claim.schema.indicator):
                    memory.record_stat_id_used(
                        indicator=claim.schema.indicator,
                        stat_id=evidence.stat_table_id,
                        category_path=evidence.category_path,
                        time_period=evidence.time_period,
                    )
                logger.info(f"[Agent A] Step 8 verify_claim → {result.verdict.value}")

                # Action: generate_explanation (LLM)
                # Tool: HCX-003 (v1, 중량) — verdict별 전용 프롬프트 사용
                # Thought: "판정 결과를 독자가 이해할 수 있는 설명으로 생성해야 한다"
                # Observation: 자연어 설명 문자열 + provenance_summary 세팅
                result.explanation = await generate_explanation(claim, result, self.config)
                logger.info(f"[Agent A] Step 9 generate_explanation → {len(result.explanation or '')}자")

                return result

        results = list(await asyncio.gather(*[process_one_claim(c) for c in claims]))

        # [이수민 2026-05-14] working memory 최종 통계 로깅
        logger.info(f"[Agent A] working_memory stats: {memory.stats()}")
        if memory.rejected_stat_ids:
            logger.info(f"[Agent A] 도메인 가드 거절: {len(memory.rejected_stat_ids)}건")

        logger.info(f"[Agent A] 완료: claims={len(claims)}, results={len(results)}, "
                    f"nodes={len(all_nodes)}, edges={len(all_edges)}")
        return claims, results, all_nodes, all_edges

# ── [v4 김예슬] Context Window 헬퍼 ─────────────────────────────────────────

def _get_context_window(
    claim: "Claim",
    sir_doc: "SIRDocument",
    window: int = 2,
) -> str:
    """
    claim의 앞 문장 window개를 SIR Tree에서 가져와서 context 문자열로 반환.

    [v4 김예슬 - 2026-05-07]
    "이는 20년 새 2.6배 증가한 것이다" 같은 문장은
    앞 문장 "2024년 쉬었음 청년이 21만7천명이다"가 있어야 정확한 schema 추출 가능.

    SIR Tree의 block_id/sent_id를 기반으로 같은 블록 내 앞 문장들을 찾음.
    NEXT_SENT 엣지를 graph에서 탐색하지 않고 sir_doc에서 직접 조회 (더 효율적).

    Args:
        claim: 대상 주장
        sir_doc: SIR 문서 (blocks → sentences 계층)
        window: 앞에서 가져올 문장 수 (기본 2)

    Returns:
        "앞문장1. 앞문장2. 현재문장" 형태의 context 문자열
    """
    target_block = claim.block_id
    target_sent  = claim.sent_id

    # sir_doc에서 해당 블록 찾기
    block = None
    for b in sir_doc.blocks:
        if b.block_id == target_block:
            block = b
            break

    if not block or not block.sentences:
        return claim.claim_text

    # 현재 문장 인덱스 찾기
    sent_idx = None
    for i, sent in enumerate(block.sentences):
        if sent.sent_id == target_sent:
            sent_idx = i
            break

    if sent_idx is None:
        return claim.claim_text

    # 앞 window개 문장 수집
    start = max(0, sent_idx - window)
    context_sents = []
    for i in range(start, sent_idx + 1):
        text = block.sentences[i].text.strip()
        if text:
            context_sents.append(text)

    return " ".join(context_sents)