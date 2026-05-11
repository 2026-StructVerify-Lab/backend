"""
tests/test_step3_5_detection.py — Step 3~5 Detection 파이프라인 테스트

실행 방법:
    # API 키 없이 되는 단위 테스트
    python -m pytest tests/test_step3_5_detection.py -k "unit" -v

    # NCP 키 필요 (실제 LLM 호출)
    export NCP_API_KEY="nv-xxx"
    python -m pytest tests/test_step3_5_detection.py -k "llm" -v -s
"""
import os
import tempfile
import pytest
from pathlib import Path

pytestmark = pytest.mark.asyncio

HAS_OPENAI_KEY = bool(os.environ.get("NCP_API_KEY"))
skip_no_openai = pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY 없음")

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

@skip_no_openai
async def test_url_step1_to_6(llm_config):
    assert TEST_URL, "TEST_URL 환경변수를 설정해야 함"

    from structverify.core.schemas import SourceType
    from structverify.preprocessing.extractor import extract_text
    from structverify.preprocessing.sir_builder import build_sir
    from structverify.detection.domain_classifier import classify_domain
    from structverify.detection.claim_detector import detect_claims
    from structverify.detection.schema_inductor import induce_schemas
    from structverify.graph.graph_builder import build_claim_graph
    from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
    from structverify.verification.verifier import verify_claim
    config = {
        "llm": llm_config,
        "domain_registry_path": f"{tempfile.mkdtemp()}/registry.yaml",
        "candidate_detection": {
            "threshold": 0.5,
            "teacher_llm_fallback": True,
        },
        "verification": {
            "min_confidence": 0.5,
        },
    }

    print(f"\n[Step 1] URL 입력: {TEST_URL}")

    raw_text = await extract_text(TEST_URL, SourceType.URL)

    assert isinstance(raw_text, str)
    assert len(raw_text.strip()) > 50

    print(f"[Step 1] 추출 텍스트 길이: {len(raw_text)}")
    print(raw_text[:300].replace("\n", " "))

    print("\n[Step 2] SIR Tree 생성")

    sir_doc = build_sir(
        raw_text,
        SourceType.URL,
        source_uri=TEST_URL,
    )

    assert sir_doc is not None
    assert len(sir_doc.blocks) > 0

    print(f"[Step 2] blocks={len(sir_doc.blocks)}")

    print("\n[Step 3] 도메인 분류")

    domain, description = await classify_domain(sir_doc, config)
    config["detected_domain"] = domain

    assert isinstance(domain, str)
    assert len(domain) > 0

    print(f"[Step 3] domain={domain}")
    print(f"[Step 3] description={description}")

    print("\n[Step 4] Claim Detection")

    claims = await detect_claims(sir_doc, config)

    assert isinstance(claims, list)

    print(f"[Step 4] claims={len(claims)}")

    for claim in claims[:5]:
        print(
            f"- [{claim.sent_id}] "
            f"score={claim.check_worthy_score:.2f} | "
            f"{claim.claim_text[:100]}"
        )

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
    print("\n[Step 6-1] Graph Visualization")

    from pyvis.network import Network

    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)

    html_path = out_dir / "claim_graph.html"

    net = Network(
        height="850px",
        width="100%",
        directed=True,
        notebook=False,
    )

    # 노드 추가
    for node in nodes:
        node_type = getattr(node.node_type, "value", str(node.node_type))
        label = node.label or node.node_id

        net.add_node(
            node.node_id,
            label=label[:40],
            title=f"""
ID: {node.node_id}
TYPE: {node_type}
LABEL: {label}
PROPERTIES: {node.properties}
""",
            group=node_type,
        )

    # 엣지 추가
    for edge in edges:
        edge_type = getattr(edge.edge_type, "value", str(edge.edge_type))

        net.add_edge(
            edge.from_node,
            edge.to_node,
            label=edge_type,
            title=f"""
TYPE: {edge_type}
PROPERTIES: {edge.properties}
""",
            arrows="to",
        )

    net.toggle_physics(True)
    net.show_buttons(filter_=["physics"])

    net.write_html(str(html_path))

    print(f"[Graph HTML] {html_path.resolve()}")
    
    
    
    print("\n[Step 7] Evidence Retrieval + Subgraph")

    evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
        self.kosis, query, claim_nid
    )

    assert isinstance(evidence_nodes, list)
    assert isinstance(evidence_edges, list)

    nodes.extend(evidence_nodes)
    edges.extend(evidence_edges)

    print(f"[Step 7] evidence_nodes={len(evidence_nodes)}, evidence_edges={len(evidence_edges)}")
    print(f"[Step 7] total_nodes={len(nodes)}, total_edges={len(edges)}")


    print("\n[Step 8] Verification")

    from structverify.verification.verifier import verify_claims

    result = verify_claim(claim, evidence, config)

    assert isinstance(verification_results, list)

    for r in verification_results:
        print({
            "claim_id": getattr(r, "claim_id", None),
            "verdict": getattr(r, "verdict", None),
            "confidence": getattr(r, "confidence", None),
        })


    print("\n[Step 9] Explanation + Provenance")

    from structverify.explanation.explainer import generate_explanations

    explanations = await generate_explanations(
        claims=claims,
        verification_results=verification_results,
        graph_nodes=nodes,
        graph_edges=edges,
        config=config,
    )

    assert isinstance(explanations, list)

    for e in explanations:
        print({
            "claim_id": getattr(e, "claim_id", None),
            "summary": getattr(e, "summary", None),
            "provenance": getattr(e, "provenance", None),
        })
        
        
    print("\n[Step 9-1] Full Pipeline JSON Export")

    import json
    from pathlib import Path

    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "step1_9_pipeline_result.json"

    result_json = {
        "summary": {
            "domain": domain,
            "description": description,
            "claim_count": len(claims),
            "node_count": len(nodes),
            "edge_count": len(edges),
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
                "canonical_type": (
                    claim.canonical_type.value
                    if hasattr(claim.canonical_type, "value")
                    else claim.canonical_type
                ),
                "score": claim.check_worthy_score,
                "schema": claim.schema.model_dump() if claim.schema else None,
            }
            for claim in claims
        ],
        "graph": {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": getattr(node.node_type, "value", str(node.node_type)),
                    "label": node.label,
                    "properties": node.properties or {},
                }
                for node in nodes
            ],
            "edges": [
                {
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "edge_type": getattr(edge.edge_type, "value", str(edge.edge_type)),
                    "properties": edge.properties or {},
                }
                for edge in edges
            ],
        },
        "verification_results": [
            r.model_dump() if hasattr(r, "model_dump") else str(r)
            for r in verification_results
        ],
        "explanations": [
            e.model_dump() if hasattr(e, "model_dump") else str(e)
            for e in explanations
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"[Full Pipeline JSON] {json_path.resolve()}")