"""
agent/runtime_agent.py — Runtime Verification Agent (Agent A)

실시간 검증 요청을 처리하는 메인 Agent. ReAct 패턴 기반.
Thought → Action(Tool Call) → Observation 순환을 통해 파이프라인 제어.

[김예슬]
- Step 3~9 오케스트레이션 전체 담당
- ReAct 패턴으로 각 스텝을 Action으로 정의하고 순차 실행
- 에러 복구 및 재시도 로직 추가 예정

[ReAct 패턴 설명]
  LLM이 단순히 답변을 생성하는 것이 아니라, 매 스텝마다 다음을 반복합니다:
    Thought  : "현재 상태에서 무엇을 해야 하는가?" (LLM 내부 추론)
    Action   : 구체적인 도구(함수) 호출 — classify_domain, score_candidate 등
    Observation: 도구 호출 결과를 관찰하고 다음 Thought 수행

  이 Agent에서 각 step은 하나의 Action에 대응합니다:
    Action: classify_domain   → Tool: LLM 분류기 (HCX-DASH)
    Action: score_candidate   → Tool: Teacher LLM (HCX-DASH)
    Action: check_worthiness  → Tool: LLM 주장 판별기 (HCX-003)
    Action: induce_schemas    → Tool: LLM 스키마 추출기 (HCX-003)
    Action: build_graph       → Tool: 내부 그래프 빌더 (LLM 미사용)
    Action: retrieve_evidence → Tool: KOSIS Open API
    Action: verify_claim      → Tool: Deterministic 수치 비교 엔진 (LLM 미사용)
    Action: generate_explain  → Tool: LLM 설명 생성기 (HCX-003)

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
  LLM이 사고와 행동을 교차하며 문제를 해결하는 프레임워크
"""
from __future__ import annotations
from structverify.core.schemas import (
    Claim, SIRDocument, VerificationResult, GraphNode, GraphEdge)
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

    각 스텝이 하나의 Action에 해당합니다:
      Step 3  → Action: classify_domain    (LLM Tool)
      Step 4a → Action: score_candidate    (Teacher LLM Tool)
      Step 4b → Action: check_worthiness   (LLM Tool)
      Step 5  → Action: induce_schemas     (LLM Tool)
      Step 6  → Action: build_claim_graph  (내부 로직)
      Step 7  → Action: retrieve_evidence  (KOSIS API Tool)
      Step 8  → Action: verify_claim       (Deterministic, LLM 미사용)
      Step 9  → Action: generate_explain   (LLM Tool)
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.kosis = KOSISConnector(config=self.config.get("kosis", {}))

        # TODO [김예슬]: Graph Store 초기화 (Neo4j 노드/엣지 실시간 저장용)
        # from structverify.graph.graph_store import GraphStore
        # self.graph_store = GraphStore(config=self.config.get("graph", {}))

        # TODO [김예슬]: Domain Pack 레지스트리 로드
        #   domain-packs/{domain}/prompts.yaml 파일을 미리 로드해서
        #   각 스텝의 프롬프트에 도메인별 few-shot 예시를 주입

    async def process(self, sir_doc: SIRDocument) -> tuple[
        list[Claim], list[VerificationResult], list[GraphNode], list[GraphEdge]
    ]:
        """
        SIR 문서를 입력받아 전체 검증 파이프라인을 실행한다.

        ReAct 흐름:
          Thought: "도메인을 먼저 파악해야 한다"
          Action (Step 3): classify_domain
          Observation: domain 확인

          Thought: "검증 가능한 주장을 찾아야 한다"
          Action (Step 4): detect_claims (내부적으로 candidate scoring → check-worthiness)
          Observation: claims 목록 확인

          Thought: "각 주장을 구조화해야 한다"
          Action (Step 5): induce_schemas
          Observation: 각 claim에 ClaimSchema 추가됨

          Thought: "그래프를 구성해야 한다"
          Action (Step 6): build_claim_graph
          Observation: nodes, edges 생성됨

          [각 주장에 대해 반복]
          Thought: "KOSIS에서 공식 수치를 가져와야 한다"
          Action (Step 7): build_evidence_subgraph (KOSIS API 호출)
          Observation: Evidence 확보

          Thought: "수치를 비교하여 판정해야 한다"
          Action (Step 8): verify_claim (Deterministic)
          Observation: verdict, confidence 확인

          Thought: "자연어 설명을 생성해야 한다"
          Action (Step 9): generate_explanation (LLM)
          Observation: explanation 텍스트 생성됨

        Returns:
            (claims, results, graph_nodes, graph_edges)
        """
        # ── Action: classify_domain ──────────────────────────────────
        # TODO [김예슬]: domain_classifier.py 프롬프트 설계 및 튜닝
        #   - few-shot 예시 추가 (domain-packs 활용)
        #   - 신뢰도 낮을 때 "general" 도메인으로 fallback
        # domain = await classify_domain(sir_doc, self.config)
        domain, domain_desc = await classify_domain(sir_doc, self.config) #[김예슬] - 도메인 설명도 반환하도록 수정
        self.config["detected_domain"] = domain  
        logger.info(f"[Agent A] Action: classify_domain → Observation: {domain} ({domain_desc})")

        # ── Action: detect_claims (candidate scoring + check-worthiness) ──
        # TODO [김예슬]: candidate_scorer.py — Teacher LLM 점수화 로직 완성
        #   - 학습 데이터 누적 후 경량 분류 모델로 교체 계획
        # TODO [김예슬]: claim_detector.py — check-worthiness 프롬프트 튜닝
        #   - domain-packs의 도메인별 예시 주입
        claims = await detect_claims(sir_doc, self.config)
        logger.info(f"[Agent A] Action: detect_claims → Observation: {len(claims)} claims")

        # ── Action: induce_schemas ────────────────────────────────────
        # TODO [김예슬]: schema_inductor.py — JSON 구조화 프롬프트 설계
        #   - indicator / time_period / unit / population / value 추출
        #   - 실패 시 partial schema라도 반환 (None 대신)
        claims = await induce_schemas(claims, self.config)
        logger.info(f"[Agent A] Action: induce_schemas → Observation: schemas attached")

        # ── Action: build_claim_graph ─────────────────────────────────
        # TODO [신준수]: graph_builder.py — 노드/엣지 타입 및 관계 완성
        all_nodes, all_edges = build_claim_graph(claims)
        logger.info(f"[Agent A] Action: build_claim_graph → Observation: {len(all_nodes)} nodes")

        # ── 각 주장별 Step 7~9 ────────────────────────────────────────
        results: list[VerificationResult] = []
        for claim in claims:
            query = build_query(claim)
            claim_nid = f"claim:{claim.claim_id.hex[:8]}"

            # Action: retrieve_evidence (KOSIS API)
            # TODO [신준수]: kosis_connector.py — 실제 HTTP 호출 구현
            # TODO [신준수]: query_builder.py — ClaimSchema → KOSIS 파라미터 변환
            evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
                self.kosis, query, claim_nid)
            all_nodes.extend(ev_nodes)
            all_edges.extend(ev_edges)
            logger.info(f"[Agent A] Action: retrieve_evidence → Observation: {evidence}")

            # Action: verify_claim (Deterministic, LLM 미사용)
            # TODO [신준수]: verifier.py — TIME_PERIOD/POPULATION/EXAGGERATION 세분화
            result = verify_claim(claim, evidence, self.config)
            logger.info(f"[Agent A] Action: verify_claim → Observation: {result.verdict.value}")

            # Action: generate_explanation (LLM)
            # TODO [김예슬]: explainer.py — 설명 생성 프롬프트 설계
            #   - verdict 유형별 다른 설명 템플릿 적용
            #   - provenance 정보 포함 (출처 URL, 표 ID 등)
            result.explanation = await generate_explanation(claim, result, self.config)
            results.append(result)

        logger.info(f"[Agent A] 완료: {len(claims)} claims, {len(results)} results")
        return claims, results, all_nodes, all_edges
