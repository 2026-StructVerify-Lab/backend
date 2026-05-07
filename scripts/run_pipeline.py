import asyncio
import json
from pathlib import Path
from structverify.core.pipeline import VerificationPipeline

async def main():
    pipeline = VerificationPipeline()

    report = await pipeline.run(
        "https://www.yna.co.kr/view/AKR20260420071900003",
        "url"
    )

    # 터미널 출력
    print(f"\n도메인: {report.domain_pack_used}")
    print(f"claims: {len(report.claims)}, results: {len(report.results)}")
    print(f"nodes: {len(report.graph_nodes)}, edges: {len(report.graph_edges)}\n")

    verdict_count = {}
    for r in report.results:
        v = r.verdict.value
        verdict_count[v] = verdict_count.get(v, 0) + 1
        print(f"  [{v}] {r.explanation[:100] if r.explanation else ''}")

    print(f"\n판정 분포: {verdict_count}")

    # JSON 저장
    out = Path("test_outputs")
    out.mkdir(exist_ok=True)

    def _ev(v):
        return v.value if hasattr(v, "value") else v

    result_json = {
        "domain": report.domain_pack_used,
        "claim_count": len(report.claims),
        "verdict_distribution": verdict_count,
        "results": [
            {
                "claim_id":    str(r.claim_id),
                "verdict":     _ev(r.verdict),
                "confidence":  r.confidence,
                "explanation": r.explanation,
            }
            for r in report.results
        ],
        "claims": [
            {
                "sent_id":    c.sent_id,
                "claim_text": c.claim_text,
                "indicator":  c.schema.indicator if c.schema else None,
                "value":      c.schema.value     if c.schema else None,
            }
            for c in report.claims
        ],
    }

    json_path = out / "pipeline_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n결과 저장: {json_path.resolve()}")

asyncio.run(main())