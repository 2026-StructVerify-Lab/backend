"""
임시 — 출생아 수 기사로 working memory 통합 후 파이프라인 검증.
"""
import asyncio
import json
from pathlib import Path

from structverify.core.pipeline import VerificationPipeline
from structverify.graph.claim_graph import ClaimGraph

TEXT = """# 4월 출생아 수, 34년 만에 최대 증가…혼인도 6년 만에 최고치

2025-06-25

코로나 이후 미뤘던 결혼 본격화
"결혼·출산 환경 개선 지속 필요"
[서울경제]
올해 4월 출생아 수는 총 2만 717명으로 지난해 같은 달(1만 9059명)보다 8.7% 늘었다.
이는 1991년 4월(8.7%) 이후 34년 만에 가장 높은 증가율이다.
올 4월 합계출산율도 0.79명으로 지난해 같은 달보다 0.06명 증가했다.
올 4월 혼인 건수는 1만 8921건으로 전년보다 4.9% 증가했다.
"""


async def main():
    pipeline = VerificationPipeline()
    report = await pipeline.run(TEXT, "text")

    print(f"\n{'='*78}")
    print(f"도메인: {report.domain_pack_used}")
    print(f"claims: {len(report.claims)}, results: {len(report.results)}")
    print(f"{'='*78}\n")

    # 판정 분포
    verdict_count = {}
    mismatch_count = {}
    for r in report.results:
        v = r.verdict.value
        verdict_count[v] = verdict_count.get(v, 0) + 1
        if r.mismatch_type:
            mt = r.mismatch_type.value
            mismatch_count[mt] = mismatch_count.get(mt, 0) + 1

    print(f"판정 분포: {verdict_count}")
    print(f"mismatch 유형: {mismatch_count}\n")

    # 각 claim 결과
    for idx, (claim, r) in enumerate(zip(report.claims, report.results), 1):
        v = r.verdict.value
        mt = r.mismatch_type.value if r.mismatch_type else "-"
        print(f"[{idx:02d}] [{v}/{mt}] conf={r.confidence:.2f}")
        if claim.schema:
            s = claim.schema
            print(f"     indicator={s.indicator!r} value={s.value} time={s.time_period!r}")
        if r.evidence:
            ev = r.evidence
            print(f"     evidence: {ev.stat_table_id} | category={ev.category_path!r}")
            print(f"               official={ev.official_value} {ev.unit}")
        print()

    # JSON 저장
    out = Path("test_outputs")
    out.mkdir(exist_ok=True)
    result_json = {
        "domain": report.domain_pack_used,
        "verdict_distribution": verdict_count,
        "mismatch_distribution": mismatch_count,
        "results": [
            {
                "sent_id": c.sent_id,
                "indicator": c.schema.indicator if c.schema else None,
                "value": c.schema.value if c.schema else None,
                "time_period": c.schema.time_period if c.schema else None,
                "verdict": r.verdict.value,
                "mismatch_type": r.mismatch_type.value if r.mismatch_type else None,
                "confidence": r.confidence,
                "evidence": {
                    "stat_id": r.evidence.stat_table_id,
                    "category_path": r.evidence.category_path,
                    "official_value": r.evidence.official_value,
                } if r.evidence else None,
            }
            for c, r in zip(report.claims, report.results)
        ],
    }
    json_path = out / "pipeline_birth_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"결과 저장: {json_path.resolve()}")


asyncio.run(main())
