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
from structverify.retrieval.query_builder import build_query
from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
from structverify.retrieval.kosis_connector import KOSISConnector
from structverify.verification.verifier import verify_claim
from structverify.explanation.explainer import generate_explanation
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class RuntimeAgent:
    """
    Agent A: 실시간 검증 처리.
    Step 3~9를 순차 실행하며 ReAct 패턴으로 파이프라인을 오케스트레이션한다.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.kosis = KOSISConnector(config=self.config.get("kosis", {}))

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

        # ── Action: induce_schemas ───────────────────────────────────
        # Tool: HCX-007 Structured Outputs (v3 API)
        # Thought: "각 주장을 indicator/value/unit/population으로 구조화해야 한다"
        # Observation: claim.schema = ClaimSchema({indicator, value, ...})
        # 기존 generate_json() + 파싱 실패 재시도 → Structured Outputs로 교체
        # JSON Schema 정의로 파싱 실패 자체가 없어짐
        claims = await induce_schemas(claims, self.config)
        logger.info(f"[Agent A] Step 5 induce_schemas → schemas attached")

        # ── Action: build_claim_graph ────────────────────────────────
        # Tool: 내부 로직 (LLM 미사용)
        # Thought: "ClaimSchema → Knowledge Graph 노드/엣지를 구성해야 한다"
        # Observation: GraphNode[], GraphEdge[]
        # TODO [신준수]: graph_builder.py 노드/엣지 타입 완성
        all_nodes, all_edges = build_claim_graph(claims, sir_doc=sir_doc) # 호출부 로직 변경 [pipeline v3] 김예슬
        logger.info(f"[Agent A] Step 6 build_claim_graph → {len(all_nodes)} nodes")

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
            evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
                self.kosis, query, claim_nid,
            )
            all_nodes.extend(ev_nodes)
            all_edges.extend(ev_edges)
            logger.info(f"[Agent A] Step 7 retrieve_evidence → {evidence}")

            # Action: verify_claim (Deterministic, LLM 미개입)
            # Tool: 수치 비교 엔진 — hallucination 방지를 위해 LLM 사용 안 함
            # Thought: "수치를 비교하여 MATCH/MISMATCH/UNVERIFIABLE 판정해야 한다"
            # Observation: verdict + mismatch_type + confidence
            # TODO [신준수]: verifier.py TIME_PERIOD/POPULATION/EXAGGERATION 세분화
            result = verify_claim(claim, evidence, self.config)
            logger.info(f"[Agent A] Step 8 verify_claim → {result.verdict.value}")

            # Action: generate_explanation (LLM)
            # Tool: HCX-003 (v1, 중량) — verdict별 전용 프롬프트 사용
            # Thought: "판정 결과를 독자가 이해할 수 있는 설명으로 생성해야 한다"
            # Observation: 자연어 설명 문자열 + provenance_summary 세팅
            result.explanation = await generate_explanation(claim, result, self.config)
            logger.info(f"[Agent A] Step 9 generate_explanation → {len(result.explanation or '')}자")

            results.append(result)

        logger.info(f"[Agent A] 완료: claims={len(claims)}, results={len(results)}, "
                    f"nodes={len(all_nodes)}, edges={len(all_edges)}")
        return claims, results, all_nodes, all_edges