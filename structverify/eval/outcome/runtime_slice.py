"""Eval-only orchestration mirroring RuntimeAgent.process (detect skipped).

Keep in sync with structverify/agent/runtime_agent.py process() when that changes.
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from structverify.agent.runtime_agent import RuntimeAgent, _get_context_window
from structverify.core.schemas import (
    Claim,
    GraphEdge,
    GraphNode,
    SIRDocument,
    VerificationResult,
)
from structverify.detection.domain_classifier import classify_domain
from structverify.detection.schema_inductor import induce_schemas
from structverify.graph.claim_graph import ClaimGraph
from structverify.graph.document_graph import build_document_temporal_graph
from structverify.graph.graph_builder import build_claim_graph
from structverify.memory.working_memory import DocumentWorkingMemory
from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
from structverify.retrieval.query_builder import build_query
from structverify.graph.graph_multihop import apply_multihop_verification
from structverify.verification.verifier import verify_claim
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_workspace_scope(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Per-case workspace isolation (avoid cross-case verified_facts cache)."""
    cfg = dict(config)
    eval_cfg = dict(cfg.get("eval") or {})
    if eval_cfg.get("workspace_scope") == "per_case":
        agent_cfg = dict(cfg.get("agent") or {})
        ws_cfg = dict(agent_cfg.get("workspace") or {})
        ws_cfg["scope"] = "job_id"
        ws_cfg["external_job_id"] = case_id
        agent_cfg["workspace"] = ws_cfg
        cfg["agent"] = agent_cfg
    return cfg


async def run_outcome_slice(
    sir_doc: SIRDocument,
    oracle_claims: list[Claim],
    config: dict[str, Any],
    *,
    schema_mode: str = "induce",
    case_id: str | None = None,
) -> tuple[list[Claim], list[VerificationResult]]:
    """Run domain → schema → verify for oracle claims (no detect_claims)."""
    eval_cfg = config.get("eval") or {}
    workspace_case_id = case_id or (
        str(oracle_claims[0].claim_id) if oracle_claims else "eval"
    )
    slice_config = _apply_workspace_scope(config, workspace_case_id)
    agent = RuntimeAgent(config=slice_config)
    memory = DocumentWorkingMemory(
        doc_id=str(sir_doc.doc_id),
        run_id=str(uuid4())[:8],
        source_uri=getattr(sir_doc, "source_uri", None),
    )

    domain_oracle = eval_cfg.get("domain_oracle", True)
    if domain_oracle and config.get("detected_domain"):
        domain = config["detected_domain"]
        domain_desc = config.get("detected_domain_desc") or domain
        logger.info(f"[eval] domain oracle → {domain}")
    else:
        domain, domain_desc = await classify_domain(sir_doc, config)
        config["detected_domain"] = domain
        config["detected_domain_desc"] = domain_desc
    memory.record_domain(domain, domain_desc)

    claims = list(oracle_claims)
    if not claims:
        return [], []

    for claim in claims:
        if not claim.context_text:
            claim.context_text = _get_context_window(claim, sir_doc, window=2)

    temporal_graph = None
    try:
        t_nodes, t_edges = await build_document_temporal_graph(sir_doc, config)
        if t_nodes:
            temporal_graph = ClaimGraph(t_nodes, t_edges)
            for _n in t_nodes:
                _nt = getattr(getattr(_n, "node_type", None), "value", "")
                if _nt == "document":
                    _ay = (_n.properties or {}).get("anchor_year")
                    if _ay:
                        try:
                            memory.record_anchor_year(int(_ay))
                        except (ValueError, TypeError):
                            pass
    except Exception as e:
        logger.warning(f"[eval] temporal graph failed: {e}")

    if schema_mode == "oracle":
        logger.info(f"[eval] schema_mode=oracle — skip induce_schemas ({len(claims)} claims)")
    else:
        claims = await induce_schemas(claims, config, graph=temporal_graph)
    memory.record_claims(claims)

    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    all_nodes, all_edges = build_claim_graph(claims, sir_doc=sir_doc)

    agent_enabled = bool((config.get("agent") or {}).get("enabled", False))
    source_text = (
        getattr(sir_doc, "raw_text", None)
        or agent._get_source_text(sir_doc)
    )
    anchor_year = temporal_graph.get_anchor_year() if temporal_graph else None

    sem = asyncio.Semaphore(3)
    mem_lock = asyncio.Lock()

    async def process_one_claim(claim: Claim):
        claim_nid = f"claim:{claim.claim_id.hex[:8]}"
        async with sem:
            if agent_enabled:
                return await agent._verify_with_agent(
                    claim,
                    source_text,
                    anchor_year,
                    temporal_graph,
                    claim_nid=claim_nid,
                    memory=memory,
                    mem_lock=mem_lock,
                )
            query = build_query(claim)
            evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
                agent.kosis, query, claim_nid
            )
            result = verify_claim(
                claim, evidence, slice_config, graph=temporal_graph
            )
            _ev_cat = None
            if evidence is not None:
                _ev_cat = (evidence.raw_response or {}).get("category_path")
            if _ev_cat and not memory.domain_matches_category(_ev_cat):
                from structverify.core.schemas import VerdictType

                result.verdict = VerdictType.UNVERIFIABLE
                result.confidence = min(result.confidence or 0.3, 0.3)
            return result, ev_nodes, ev_edges

    parallel = await asyncio.gather(*[process_one_claim(c) for c in claims])
    results: list[VerificationResult] = []
    for result, ev_nodes, ev_edges in parallel:
        results.append(result)
        all_nodes.extend(ev_nodes)
        all_edges.extend(ev_edges)

    results = apply_multihop_verification(
        claims, results, all_edges, config
    )
    return claims, results
