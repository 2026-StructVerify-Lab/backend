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
from structverify.retrieval.evidence_subgraph import (
    build_evidence_subgraph, build_evidence_subgraph_for_plan,
)
from structverify.retrieval.kosis_connector import KOSISConnector
from structverify.verification.verifier import verify_claim
from structverify.verification.evidence_check import check_evidence_relevance
from structverify.explanation.explainer import generate_explanation
from structverify.core.memory import DocumentMemory  # [v6.4 추가]
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

        # ── Action: classify_domain ──────────────────────────────────
        # Tool: HCX-DASH-002 (v3 API, 경량)
        # Thought: "이 문서의 도메인이 무엇인가?"
        # Observation: domain 문자열 + 설명
        domain, domain_desc = await classify_domain(sir_doc, self.config)
        self.config["detected_domain"] = domain
        self.config["detected_domain_desc"] = domain_desc
        logger.info(f"[Agent A] Step 3 classify_domain → {domain} ({domain_desc})")

        # [v6.4 추가] 문서 단위 작업 기억 — 이 process() 호출 동안만 유효
        # claim 간 컨텍스트 공유 + KOSIS 표 캐싱 + indicator → stat_id 매핑
        memory = DocumentMemory()
        logger.info(f"[Agent A] DocumentMemory 초기화")

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
        # [v6.4 추가] memory 전달 → 이전 claim들의 schema/plan을 prompt에 컨텍스트 주입
        claims = await induce_schemas(
            claims, self.config, graph=pre_graph, memory=memory,
        )
        logger.info(f"[Agent A] Step 5 induce_schemas → schemas attached")

        # ── Action: build_claim_graph ────────────────────────────────
        # [v6] 기존 ClaimNode/MetricNode/EntityNode/COMPARE 생성
        all_nodes, all_edges = build_claim_graph(claims, sir_doc=sir_doc)
        # document temporal 그래프 합치기
        all_nodes.extend(doc_nodes)
        all_edges.extend(doc_edges)
        logger.info(f"[Agent A] Step 6 build_claim_graph → {len(all_nodes)} nodes")

        # [v6] 전체 그래프로 ClaimGraph 재생성 (verifier/explainer가 사용)
        full_graph = ClaimGraph(all_nodes, all_edges)

        # ── 각 주장별 Step 7~9 ──────────────────────────────────────
        results: list[VerificationResult] = []

        for claim in claims:
            claim_nid = f"claim:{claim.claim_id.hex[:8]}"

            # Action: retrieve_evidence (KOSIS API)
            # Tool: KOSIS Open API — pgvector 검색 → LLM 리랭킹 → 실제 수치 조회
            # Thought: "KOSIS에서 공식 수치를 가져와야 한다"
            # Observation: Evidence {official_value, stat_table_id, ...}
            # TODO [신준수]: kosis_connector.py 실제 HTTP 호출 구현
            # TODO [신준수]: query_builder.py ClaimSchema → KOSIS 파라미터 변환
            query = build_query(claim)

            # ── [v6.3] Step 7: plan-aware retrieval ─────────────────────
            # claim.schema.evidence_plan에 따라 1~2개 evidence를 가져옴.
            # measurement → 1개, delta/ratio → 2개 (같은 통계표 시점만 변경)
            plan = claim.schema.evidence_plan if claim.schema else None
            evidences: list = []
            evidence = None  # primary evidence (하위 호환)
            ev_nodes: list = []
            ev_edges: list = []

            if plan is not None and plan.requirements:
                evidences, ev_nodes, ev_edges = await build_evidence_subgraph_for_plan(
                    self.kosis, query, plan, claim_nid, memory=memory,  # [v6.4 추가]
                )
                if evidences:
                    evidence = next(
                        (e for e in evidences if e.requirement_role in ("primary", "endpoint_a")),
                        evidences[0],
                    )

            # [v6.3.1] plan 기반 조회가 비어있으면 단일 fallback 시도
            # — anchor 시점이 너무 구체적이거나 KOSIS 데이터 부재로 실패한 경우
            # — 단일 evidence라도 가져와야 partial verification 가능
            if not evidences:
                if plan is not None and plan.requirements:
                    logger.info(
                        f"[Agent A] Step 7 plan-fetch 실패 → 단일 fallback 시도"
                    )
                evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
                    self.kosis, query, claim_nid, memory=memory,  # [v6.4 추가]
                )
                if evidence:
                    evidences = [evidence]

            all_nodes.extend(ev_nodes)
            all_edges.extend(ev_edges)
            logger.info(
                f"[Agent A] Step 7 retrieve_evidence → "
                f"{len(evidences)}개 evidence "
                f"(plan_combiner={plan.combiner if plan else 'none'})"
            )

            # ── [v6.2] Step 7.5: evidence relevance check (LLM 1회) ──────
            # plan의 첫 번째(anchor) evidence만 검증 — 같은 표에서 가져온
            # 나머지 endpoint는 자동으로 동일 의미.
            if evidence is not None:
                is_relevant, reason = await check_evidence_relevance(
                    claim, evidence, self.config,
                )
                if not is_relevant:
                    logger.info(
                        f"[Agent A] Step 7.5 evidence 무관 → discard: "
                        f"[{evidence.stat_table_id}] {evidence.source_name} | {reason}"
                    )
                    evidence = None
                    evidences = []
                else:
                    logger.debug(f"[Agent A] Step 7.5 evidence 통과 | {reason}")

            # Action: verify_claim (Deterministic, LLM 미개입)
            # [v6.3] evidences(list) 전달 → combiner 분기로 delta/ratio 검증 가능
            result = verify_claim(
                claim, evidence, self.config,
                graph=full_graph, evidences=evidences,
            )
            logger.info(f"[Agent A] Step 8 verify_claim → {result.verdict.value}")

            # Action: generate_explanation (LLM)
            # Tool: HCX-003 (v1, 중량) — verdict별 전용 프롬프트 사용
            # Thought: "판정 결과를 독자가 이해할 수 있는 설명으로 생성해야 한다"
            # Observation: 자연어 설명 문자열 + provenance_summary 세팅
            result.explanation = await generate_explanation(claim, result, self.config)
            logger.info(f"[Agent A] Step 9 generate_explanation → {len(result.explanation or '')}자")

            # [v6.4 추가] 이 claim 처리 결과를 memory에 반영
            # schema_inductor에서 이미 memo append 했음 — 여기선 verdict/stat_id/계산값 보강
            if memory.processed_claims and memory.processed_claims[-1].sent_id == claim.sent_id:
                last_memo = memory.processed_claims[-1]
                last_memo.verdict = result.verdict.value
                if evidence is not None:
                    last_memo.evidence_stat_id = evidence.stat_table_id
                    if last_memo.indicator and evidence.stat_table_id:
                        memory.last_stat_for_indicator[last_memo.indicator] = evidence.stat_table_id
                # [v6.4] 계산값과 endpoint 값 보관 — 다음 claim이 "이는"으로 참조 가능
                if result.computed_value is not None:
                    last_memo.computed_value = result.computed_value
                if len(evidences) >= 2:
                    ep_a = next(
                        (e for e in evidences if e.requirement_role == "endpoint_a"),
                        None,
                    )
                    ep_b = next(
                        (e for e in evidences if e.requirement_role == "endpoint_b"),
                        None,
                    )
                    if ep_a:
                        last_memo.endpoint_a_value = ep_a.official_value
                    if ep_b:
                        last_memo.endpoint_b_value = ep_b.official_value

            results.append(result)

        logger.info(f"[Agent A] 완료: {memory.summary()} | "
                    f"claims={len(claims)}, results={len(results)}, "
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