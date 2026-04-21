"""
agent/runtime_agent.py — Runtime Verification Agent (Agent A)

실시간 검증 요청을 처리하는 메인 Agent. ReAct 패턴 기반.
Thought → Action(Tool Call) → Observation 순환을 통해 파이프라인 제어.

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
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.kosis = KOSISConnector(config=self.config.get("kosis", {}))
        # TODO: Graph Store 초기화
        # TODO: Domain Pack 레지스트리 로드

    async def process(self, sir_doc: SIRDocument) -> tuple[
        list[Claim], list[VerificationResult], list[GraphNode], list[GraphEdge]
    ]:
        """
        SIR 문서를 입력받아 전체 검증 파이프라인을 실행한다.

        Returns:
            (claims, results, graph_nodes, graph_edges)
        """
        # Step 3: 도메인 판별
        domain = await classify_domain(sir_doc, self.config)
        logger.info(f"[Agent A] 도메인: {domain}")

        # Step 4: Claim Detection
        claims = await detect_claims(sir_doc, self.config)

        # Step 5: Dynamic Schema Induction
        claims = await induce_schemas(claims, self.config)

        # Step 6: Graph Construction
        all_nodes, all_edges = build_claim_graph(claims)

        # Step 7~9: 각 주장 검증
        results: list[VerificationResult] = []
        for claim in claims:
            query = build_query(claim)
            claim_nid = f"claim:{claim.claim_id.hex[:8]}"

            # Step 7: Evidence 서브그래프
            evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
                self.kosis, query, claim_nid)
            all_nodes.extend(ev_nodes)
            all_edges.extend(ev_edges)

            # Step 8: Verification
            result = verify_claim(claim, evidence, self.config)

            # Step 9: Explanation
            result.explanation = await generate_explanation(claim, result, self.config)
            results.append(result)

        logger.info(f"[Agent A] 완료: {len(claims)} claims, {len(results)} results")
        return claims, results, all_nodes, all_edges
