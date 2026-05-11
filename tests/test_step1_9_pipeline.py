import os
import json
import tempfile
import pytest
from pathlib import Path

pytestmark = pytest.mark.asyncio

HAS_NCP_KEY = bool(os.environ.get("NCP_API_KEY"))
skip_no_ncp = pytest.mark.skipif(not HAS_NCP_KEY, reason="NCP_API_KEY 없음")

TEST_URL = os.environ.get("TEST_URL")


@pytest.fixture
def llm_config():
    return {
        "provider": "hcx",
        "models": {
            "heavy": "HCX-003",
            "light": "HCX-DASH-002",
            "structured": "HCX-007",
            "reasoning": "HCX-003",
        },
        "temperature": 0.1,
        "max_tokens": 2048,
        "openai_key_env": "NCP_API_KEY",
    }


def _enum_value(v):
    return v.value if hasattr(v, "value") else v


@skip_no_ncp
async def test_url_step1_to_9(llm_config):
    assert TEST_URL, "TEST_URL 환경변수를 설정해야 함"

    from structverify.core.schemas import SourceType
    from structverify.preprocessing.extractor import extract_text
    from structverify.preprocessing.sir_builder import build_sir
    from structverify.detection.domain_classifier import classify_domain
    from structverify.detection.claim_detector import detect_claims
    from structverify.detection.schema_inductor import induce_schemas
    from structverify.graph.graph_builder import build_claim_graph

    from structverify.retrieval.query_builder import build_query
    from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
    from structverify.retrieval.kosis_connector import KOSISConnector
    from structverify.verification.verifier import verify_claim
    from structverify.explanation.explainer import generate_explanation

    config = {
        "llm": llm_config,
        "domain_registry_path": f"{tempfile.mkdtemp()}/registry.yaml",
        "candidate_detection": {
            "threshold": 0.5,
            "teacher_llm_fallback": True,
            "concurrency": 5,
        },
        "verification": {
            "min_confidence": 0.5,
        },
    }

    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)

    print(f"\n[Step 1] URL 입력: {TEST_URL}")
    raw_text = await extract_text(TEST_URL, SourceType.URL)
    assert isinstance(raw_text, str)
    assert len(raw_text.strip()) > 50
    print(f"[Step 1] 추출 텍스트 길이: {len(raw_text)}")
    print(raw_text[:300].replace("\n", " "))

    print("\n[Step 2] SIR Tree 생성")
    sir_doc = build_sir(raw_text, SourceType.URL, source_uri=TEST_URL)
    assert sir_doc is not None
    assert len(sir_doc.blocks) > 0
    print(f"[Step 2] blocks={len(sir_doc.blocks)}")

    print("\n[Step 3] 도메인 분류")
    domain, description = await classify_domain(sir_doc, config)
    config["detected_domain"] = domain
    config["detected_domain_desc"] = description
    print(f"[Step 3] domain={domain}")
    print(f"[Step 3] description={description}")

    print("\n[Step 4] Claim Detection")
    claims = await detect_claims(sir_doc, config)
    assert isinstance(claims, list)
    print(f"[Step 4] claims={len(claims)}")
    for claim in claims[:5]:
        print(f"- [{claim.sent_id}] score={claim.check_worthy_score:.2f} | {claim.claim_text[:100]}")

    print("\n[Step 5] Schema Induction")
    if claims:
        claims = await induce_schemas(claims, config)

    for claim in claims[:5]:
        schema = claim.schema
        print({
            "indicator": getattr(schema, "indicator", None),
            "time_period": getattr(schema, "time_period", None),
            "unit": getattr(schema, "unit", None),
            "value": getattr(schema, "value", None),
            "population": getattr(schema, "population", None),
        })

    print("\n[Step 6] Graph Construction")
    nodes, edges = build_claim_graph(claims, sir_doc=sir_doc)
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    print(f"[Step 6] nodes={len(nodes)}, edges={len(edges)}")

    print("\n[Step 7] Evidence Retrieval + Subgraph")
    kosis = KOSISConnector(config=config.get("kosis", {}))

    evidence_pairs = []
    for claim in claims:
        claim_nid = f"claim:{claim.claim_id.hex[:8]}"
        query = build_query(claim)

        evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
            kosis,
            query,
            claim_nid,
        )

        nodes.extend(ev_nodes)
        edges.extend(ev_edges)
        evidence_pairs.append((claim, evidence))

        print({
            "claim_id": str(claim.claim_id),
            "claim": claim.claim_text[:60],
            "evidence": evidence,
            "ev_nodes": len(ev_nodes),
            "ev_edges": len(ev_edges),
        })

    print(f"[Step 7] total_nodes={len(nodes)}, total_edges={len(edges)}")

    print("\n[Step 8] Verification")
    verification_results = []

    for claim, evidence in evidence_pairs:
        result = verify_claim(claim, evidence, config)
        verification_results.append(result)

        print({
            "claim_id": str(claim.claim_id),
            "verdict": _enum_value(getattr(result, "verdict", None)),
            "confidence": getattr(result, "confidence", None),
        })

    print("\n[Step 9] Explanation + Provenance")
    explanations = []

    for claim, result in zip(claims, verification_results):
        explanation = await generate_explanation(claim, result, config)
        result.explanation = explanation
        explanations.append({
            "claim_id": str(claim.claim_id),
            "explanation": explanation,
        })

        print({
            "claim_id": str(claim.claim_id),
            "explanation": explanation[:120] if explanation else None,
        })

    print("\n[Step 9-1] Graph HTML Export")
    try:
        from pyvis.network import Network

        html_path = out_dir / "claim_graph_step1_9.html"
        net = Network(height="850px", width="100%", directed=True, notebook=False)

        for node in nodes:
            node_type = _enum_value(node.node_type)
            label = node.label or node.node_id
            net.add_node(
                node.node_id,
                label=label[:40],
                title=f"ID: {node.node_id}\nTYPE: {node_type}\nLABEL: {label}\nPROPERTIES: {node.properties}",
                group=node_type,
            )

        for edge in edges:
            edge_type = _enum_value(edge.edge_type)
            net.add_edge(
                edge.from_node,
                edge.to_node,
                label=edge_type,
                title=f"TYPE: {edge_type}\nPROPERTIES: {edge.properties}",
                arrows="to",
            )

        net.toggle_physics(True)
        net.show_buttons(filter_=["physics"])
        net.write_html(str(html_path))
        print(f"[Graph HTML] {html_path.resolve()}")
    except ImportError:
        print("[Graph HTML] pyvis 미설치로 skip: pip install pyvis")

    print("\n[Step 9-2] Full Pipeline JSON Export")
    json_path = out_dir / "step1_9_pipeline_result.json"

    result_json = {
        "summary": {
            "url": TEST_URL,
            "domain": domain,
            "description": description,
            "claim_count": len(claims),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "evidence_count": len(evidence_pairs),
            "verification_count": len(verification_results),
            "explanation_count": len(explanations),
        },
        "claims": [
            {
                "claim_id": str(claim.claim_id),
                "block_id": claim.block_id,
                "sent_id": claim.sent_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "canonical_type": _enum_value(claim.canonical_type),
                "score": claim.check_worthy_score,
                "schema": claim.schema.model_dump() if claim.schema else None,
            }
            for claim in claims
        ],
        "evidence": [
            {
                "claim_id": str(claim.claim_id),
                "claim_text": claim.claim_text,
                "evidence": evidence.model_dump() if hasattr(evidence, "model_dump") else evidence,
            }
            for claim, evidence in evidence_pairs
        ],
        "verification_results": [
            r.model_dump() if hasattr(r, "model_dump") else str(r)
            for r in verification_results
        ],
        "explanations": explanations,
        "graph": {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": _enum_value(node.node_type),
                    "label": node.label,
                    "properties": node.properties or {},
                }
                for node in nodes
            ],
            "edges": [
                {
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "edge_type": _enum_value(edge.edge_type),
                    "properties": edge.properties or {},
                }
                for edge in edges
            ],
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"[Full Pipeline JSON] {json_path.resolve()}")