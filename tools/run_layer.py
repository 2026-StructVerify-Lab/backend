from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from structverify.core.config_loader import load_config
from structverify.core.schemas import SourceType, Claim, ClaimSchema, SourceOffset
from structverify.preprocessing.extractor import extract_text
from structverify.preprocessing.sir_builder import build_sir
from structverify.detection.domain_classifier import classify_domain
from structverify.detection.claim_detector import detect_claims
from structverify.detection.schema_inductor import induce_schemas
from structverify.graph.graph_builder import build_claim_graph
from structverify.retrieval.query_builder import build_query
from structverify.retrieval.kosis_connector import KOSISConnector
from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
from structverify.verification.verifier import verify_claim
from structverify.explanation.explainer import generate_explanation
from tools.common import save_json, serialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a specific StructVerify layer and optionally save JSON.")
    parser.add_argument("layer", choices=[
        "extract", "sir", "domain", "claims", "schema", "graph", "retrieval", "verify", "explain", "pipeline"
    ])
    parser.add_argument("--text", type=str, help="Inline text input")
    parser.add_argument("--input-file", type=str, help="Path to input file")
    parser.add_argument("--source-type", type=str, default="text", choices=["text", "url", "pdf", "docx"])
    parser.add_argument("--config", type=str, default=os.getenv("SV_CONFIG_PATH", "config/default.yaml"))
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--claim-text", type=str, help="Manual claim text for verify/explain quick tests")
    parser.add_argument("--claimed-value", type=float, help="Manual claimed value for verify/explain quick tests")
    parser.add_argument("--official-value", type=float, help="Manual official value for verify/explain quick tests")
    return parser.parse_args()


def _resolve_source(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8") if args.source_type == "text" else args.input_file
    raise SystemExit("--text 또는 --input-file 중 하나는 필요합니다.")


async def main() -> None:
    args = parse_args()
    if args.no_save:
        os.environ["SV_SAVE_RESULTS"] = "0"
    config = load_config(args.config)
    source_type = SourceType(args.source_type)
    source = _resolve_source(args)
    output_name = args.output_name or f"layer_{args.layer}"

    raw_text = await extract_text(source, source_type)
    if args.layer == "extract":
        result = {"raw_text": raw_text, "source_type": source_type.value}
    else:
        sir_doc = build_sir(raw_text, source_type, source_uri=source if source_type == SourceType.URL else None)
        if args.layer == "sir":
            result = sir_doc
        elif args.layer == "domain":
            result = {"domain": await classify_domain(sir_doc, config), "sir_doc": sir_doc}
        elif args.layer == "claims":
            result = await detect_claims(sir_doc, config)
        elif args.layer == "schema":
            claims = await detect_claims(sir_doc, config)
            result = await induce_schemas(claims, config)
        elif args.layer == "graph":
            claims = await induce_schemas(await detect_claims(sir_doc, config), config)
            nodes, edges = build_claim_graph(claims)
            result = {"claims": claims, "graph_nodes": nodes, "graph_edges": edges}
        elif args.layer == "retrieval":
            claims = await induce_schemas(await detect_claims(sir_doc, config), config)
            if not claims:
                result = {"claims": [], "message": "No claims detected"}
            else:
                connector = KOSISConnector(config=config.get("kosis", {}))
                claim = claims[0]
                evidence, nodes, edges = await build_evidence_subgraph(connector, build_query(claim), f"claim:{claim.claim_id.hex[:8]}")
                result = {"claim": claim, "evidence": evidence, "graph_nodes": nodes, "graph_edges": edges}
        elif args.layer == "verify":
            claim = Claim(
                doc_id=uuid4(), block_id="manual", sent_id="manual",
                claim_text=args.claim_text or "manual claim",
                schema=ClaimSchema(value=args.claimed_value),
                source_offset=SourceOffset(),
            )
            from structverify.core.schemas import Evidence
            evidence = None if args.official_value is None else Evidence(source_name="manual", official_value=args.official_value)
            result = verify_claim(claim, evidence, config)
        elif args.layer == "explain":
            claim = Claim(
                doc_id=uuid4(), block_id="manual", sent_id="manual",
                claim_text=args.claim_text or "manual claim",
                schema=ClaimSchema(value=args.claimed_value),
                source_offset=SourceOffset(),
            )
            from structverify.core.schemas import Evidence
            vr = verify_claim(claim, None if args.official_value is None else Evidence(source_name="manual", official_value=args.official_value), config)
            result = {"verification": vr, "explanation": await generate_explanation(claim, vr, config)}
        elif args.layer == "pipeline":
            from structverify.core.pipeline import VerificationPipeline
            result = await VerificationPipeline(config).run(source, source_type.value)
        else:
            raise SystemExit(f"Unsupported layer: {args.layer}")

    print(json.dumps(serialize(result), ensure_ascii=False, indent=2))
    saved = save_json(result, output_name, metadata={"layer": args.layer, "source_type": source_type.value})
    if saved:
        print(f"\n[saved] {saved}")


if __name__ == "__main__":
    asyncio.run(main())
