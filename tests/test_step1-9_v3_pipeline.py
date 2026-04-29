"""
tests/test_step1_9_pipeline.py — Step 1~9 전체 파이프라인 통합 테스트 (v3)

[v3 김예슬 - 2026-04-28]
- 기본 sandbox_backend = "docker"
  · URL 입력 → trafilatura 1차 시도
  · trafilatura 실패 → LLM이 스크래핑 코드 생성 → Docker 격리 실행
  → 결과값을 Step 2~9에 그대로 연결

실행:
    # 도커 먼저 빌드 (최초 1회, 이후 _ensure_image가 자동)
    docker build -f structverify/preprocessing/Dockerfile.scraper \
                 -t structverify-scraper \
                 structverify/preprocessing/

    # 테스트 실행
    CLOVASTUDIO_API_KEY=nv-xxx \\
    KOSIS_API_KEY=xxx \\
    python -m pytest tests/test_step1_9_pipeline.py -v -s

    # URL 바꾸기 (기본: 쉬었음 청년 연합뉴스)
    TEST_URL=https://www.chosun.com/... \\
    python -m pytest tests/test_step1-9_v3_pipeline.py -v -s
"""
import os
import json
import tempfile
import pytest
from collections import Counter
from pathlib import Path

pytestmark = pytest.mark.asyncio

HAS_NCP_KEY = bool(os.environ.get("CLOVASTUDIO_API_KEY") or os.environ.get("NCP_API_KEY"))
skip_no_ncp = pytest.mark.skipif(not HAS_NCP_KEY, reason="CLOVASTUDIO_API_KEY 없음")

# 테스트할 URL (환경변수로 교체 가능)
TEST_URL = os.environ.get(
    "TEST_URL",
    "https://www.yna.co.kr/view/AKR20260420071900003"  # 기본: 쉬었음 청년 기사
)


def _enum_value(v):
    return v.value if hasattr(v, "value") else v


@pytest.fixture
def config(tmp_path):
    """파이프라인 전체 설정 — sandbox_backend 기본값 docker"""
    api_key_env = "CLOVASTUDIO_API_KEY" if os.environ.get("CLOVASTUDIO_API_KEY") else "NCP_API_KEY"
    return {
        "llm": {
            "provider":     "hcx",
            "models": {
                "heavy":      "HCX-003",
                "light":      "HCX-DASH-002",
                "structured": "HCX-007",
            },
            "temperature":  0.1,
            "max_tokens":   2048,
            "api_key_env":  api_key_env,
        },
        "kosis": {
            "api_key_env": "KOSIS_API_KEY",
            "timeout":     30,
        },
        "domain_registry_path": str(tmp_path / "registry.yaml"),
        "candidate_detection": {
            "threshold":            0.5,
            "teacher_llm_fallback": True,
            "concurrency":          2,
        },
        "verification": {
            "min_confidence": 0.5,
        },
        # [v3] 기본값 docker
        # trafilatura 실패 시 LLM 코드 생성 후 Docker 격리 실행
        "sandbox_backend":           "docker",
        # Self-Refine
        "schema_self_refine_rounds": 1,
        "explanation_self_refine":   False,  # 비용 고려 기본 off
    }


@skip_no_ncp
async def test_url_step1_to_9(config):
    """
    Step 1~9 전체 파이프라인.

    Step 1 동작 방식:
      trafilatura 성공 → 바로 Step 2 진행
      trafilatura 실패 → LLM이 사이트 전용 스크래핑 코드 생성
                      → Docker 컨테이너 격리 실행 (sandbox_backend=docker)
                      → 추출된 텍스트로 Step 2 진행

    → 어떤 URL이든 Step 2~9가 이어서 실행됨
    """
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

    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)

    # ── Step 1: URL → 텍스트 추출 ────────────────────────────────────────
    # 흐름:
    #   trafilatura 1차 시도
    #   → 성공이면 바로 반환
    #   → 실패면 LLM 스크래핑 코드 생성 → Docker 격리 실행 (sandbox_backend=docker)
    print(f"\n[Step 1] URL: {TEST_URL}")
    print(f"[Step 1] sandbox_backend={config['sandbox_backend']}")

    raw_text = await extract_text(TEST_URL, SourceType.URL)

    assert isinstance(raw_text, str), "추출 결과가 문자열이 아님"
    assert len(raw_text.strip()) > 50, f"추출된 텍스트가 너무 짧음 ({len(raw_text)}자)"

    print(f"[Step 1] 완료: {len(raw_text)}자 추출")
    print(raw_text[:200].replace("\n", " "))

    # ── Step 2: SIR Tree 변환 ────────────────────────────────────────────
    print("\n[Step 2] SIR Tree 생성")
    sir_doc = build_sir(raw_text, SourceType.URL, source_uri=TEST_URL)
    total_sents = sum(len(b.sentences) for b in sir_doc.blocks)
    print(f"[Step 2] blocks={len(sir_doc.blocks)}, sentences={total_sents}")

    assert len(sir_doc.blocks) > 0

    # ── Step 3: 도메인 분류 ──────────────────────────────────────────────
    print("\n[Step 3] 도메인 분류")
    domain, description = await classify_domain(sir_doc, config)
    config["detected_domain"]      = domain
    config["detected_domain_desc"] = description
    print(f"[Step 3] domain={domain} | {description}")

    # ── Step 4: Claim Detection ──────────────────────────────────────────
    print("\n[Step 4] Claim Detection")
    claims = await detect_claims(sir_doc, config)
    print(f"[Step 4] {len(claims)}건 탐지")
    for c in claims[:5]:
        print(f"  [{c.sent_id}] {c.check_worthy_score:.2f} | {c.claim_text[:80]}")

    # ── Step 5: Schema Induction + Self-Refine ───────────────────────────
    print(f"\n[Step 5] Schema Induction (self_refine={config['schema_self_refine_rounds']}회)")
    if claims:
        claims = await induce_schemas(claims, config)
    for c in claims[:3]:
        s = c.schema
        if s:
            print(f"  indicator={s.indicator} | value={s.value} | unit={s.unit}")

    # ── Step 6: Graph 구성 ───────────────────────────────────────────────
    print("\n[Step 6] Graph Construction")
    nodes, edges = build_claim_graph(claims, sir_doc=sir_doc)
    edge_dist = Counter(_enum_value(e.edge_type) for e in edges)
    print(f"[Step 6] nodes={len(nodes)}, edges={len(edges)}")
    print(f"  엣지 분포: {dict(edge_dist)}")

    # GraphRAG 엣지 확인
    assert "next_sent" in edge_dist, "NEXT_SENT 엣지 없음"
    assert "in_block"  in edge_dist, "IN_BLOCK 엣지 없음"
    assert "in_doc"    in edge_dist, "IN_DOC 엣지 없음"

    # ── Step 7: KOSIS Evidence 조회 ──────────────────────────────────────
    print("\n[Step 7] Evidence Retrieval")
    kosis = KOSISConnector(config=config.get("kosis", {}))
    evidence_pairs = []

    for claim in claims:
        query = build_query(claim)
        evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
            kosis, query, f"claim:{claim.claim_id.hex[:8]}"
        )
        nodes.extend(ev_nodes)
        edges.extend(ev_edges)
        evidence_pairs.append((claim, evidence))
        status = "FOUND" if evidence and getattr(evidence, "official_value", None) else "NOT FOUND"
        print(f"  [{status}] {claim.claim_text[:60]}")

    # ── Step 8: 수치 비교 판정 ───────────────────────────────────────────
    print("\n[Step 8] Verification")
    verification_results = []
    verdict_dist: Counter = Counter()

    for claim, evidence in evidence_pairs:
        result = verify_claim(claim, evidence, config)
        verification_results.append(result)
        verdict = _enum_value(result.verdict)
        verdict_dist[verdict] += 1
        print(f"  [{verdict}] {claim.claim_text[:60]}")

    print(f"  판정 분포: {dict(verdict_dist)}")

    # ── Step 9: 설명 생성 ────────────────────────────────────────────────
    print(f"\n[Step 9] Explanation (self_refine={config['explanation_self_refine']})")
    explanations = []

    for claim, result in zip(claims, verification_results):
        explanation = await generate_explanation(claim, result, config)
        result.explanation = explanation
        explanations.append({"claim_id": str(claim.claim_id), "explanation": explanation})
        verdict = _enum_value(result.verdict)
        print(f"  [{verdict}] {(explanation or '')[:100]}")

    # ── Graph HTML 시각화 ─────────────────────────────────────────────────
    print("\n[Step 9-1] Graph HTML Export")
    try:
        from pyvis.network import Network
        html_path = out_dir / "claim_graph_step1_9.html"
        net = Network(height="850px", width="100%", directed=True, notebook=False)
        for node in nodes:
            ntype = _enum_value(node.node_type)
            label = node.label or node.node_id
            net.add_node(node.node_id, label=label[:40],
                         title=f"TYPE: {ntype}\n{label}", group=ntype)
        for edge in edges:
            etype = _enum_value(edge.edge_type)
            net.add_edge(edge.from_node, edge.to_node, label=etype,
                         title=f"TYPE: {etype}", arrows="to")
        net.toggle_physics(True)
        net.show_buttons(filter_=["physics"])
        net.write_html(str(html_path))
        print(f"[Graph HTML] {html_path.resolve()}")
    except ImportError:
        print("[Graph HTML] pyvis 미설치 skip — pip install pyvis")

    # ── JSON 저장 ─────────────────────────────────────────────────────────
    print("\n[Step 9-2] Full Pipeline JSON Export")
    result_json = {
        "summary": {
            "url":             TEST_URL,
            "domain":          domain,
            "description":     description,
            "sandbox_backend": config["sandbox_backend"],
            "claim_count":     len(claims),
            "node_count":      len(nodes),
            "edge_count":      len(edges),
            "edge_distribution": dict(edge_dist),
            "verdict_distribution": dict(verdict_dist),
        },
        "claims": [
            {
                "claim_id":   str(c.claim_id),
                "block_id":   c.block_id,
                "sent_id":    c.sent_id,
                "claim_text": c.claim_text,
                "claim_type": c.claim_type,
                "score":      c.check_worthy_score,
                "schema":     c.schema.model_dump() if c.schema else None,
            }
            for c in claims
        ],
        "evidence": [
            {
                "claim_id":  str(c.claim_id),
                "claim_text": c.claim_text,
                "evidence":  ev.model_dump() if hasattr(ev, "model_dump") else ev,
            }
            for c, ev in evidence_pairs
        ],
        "verification_results": [
            r.model_dump() if hasattr(r, "model_dump") else str(r)
            for r in verification_results
        ],
        "explanations": explanations,
        "graph": {
            "nodes": [
                {"node_id": n.node_id, "node_type": _enum_value(n.node_type),
                 "label": n.label, "properties": n.properties or {}}
                for n in nodes
            ],
            "edges": [
                {"from_node": e.from_node, "to_node": e.to_node,
                 "edge_type": _enum_value(e.edge_type), "properties": e.properties or {}}
                for e in edges
            ],
        },
    }

    json_path = out_dir / "step1_9_pipeline_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"[Full Pipeline JSON] {json_path.resolve()}")

    # ── 최종 assertions ──────────────────────────────────────────────────
    assert len(verification_results) == len(claims)
    assert len(explanations) == len(claims)
    print(f"\n[완료] Step 1~9 정상 완료: claims={len(claims)}, nodes={len(nodes)}, edges={len(edges)}")