"""
임시 — 2024년 기온 기사로 working memory 통합 검증.
이전 진단에서 90% unverifiable 났던 케이스. 도메인 가드 효과 측정용.
"""
import asyncio
import json
from pathlib import Path

from structverify.core.pipeline import VerificationPipeline
from structverify.graph.claim_graph import ClaimGraph

TEXT = """# 작년 연평균기온 14.8도…사상 처음 14도 돌파 또 신기록

(서울=연합뉴스) 이재영 기자 = 2024년은 우리나라 연평균 기온이 사상 처음 14도를 돌파해 '압도적으로 뜨거웠던 해'로 남았다.
1일 기상청 기상자료개방포털을 보면 작년 평균기온은 14.8도로 재작년(13.9도)에 이어 2년 연속 '1973년 이후 연평균 기온 신기록'을 갈아치웠다.
1973년은 기상관측망이 대폭 확충돼 각종 기상기록 기준점이 되는 해다.
한 해 평균기온이 14도를 넘기는 작년이 처음이다.
작년 평균기온은 평년(1991∼2020년 평균) 연평균 기온(12.7±0.2도)을 2.3도나 웃돌았다.
일최저기온과 일최고기온 연평균 값도 지난해가 역대 1위다.
작년 평균 최저기온은 10.2도로 10도를 넘었고 평균 최고기온은 20.1도로 20도를 웃돌았다.
지난여름 기온이 41도를 기록(8월 4일 경기 여주시 점동면)한 사례가 있을 정도로 최악의 폭염이 나타났다.
특히 늦더위가 지루하게 이어진 9월은 평균기온(25.3도)이 평년기온(20.8도)보다 4.5도나 높았다.
유럽연합(EU)의 기후변화 감시 기구인 코페르니쿠스 기후변화연구소(C3S)에 따르면 작년 1∼11월 평균 지구 표면 기온이 1991∼2020년 평균보다 0.85도 높았다.
이는 아직 산업화 이래 가장 뜨거웠던 해인 재작년 같은 기간 온도보다 0.18도 높은 것이다.
"""


async def main():
    pipeline = VerificationPipeline()
    report = await pipeline.run(TEXT, "text")

    print(f"\n{'='*78}")
    print(f"도메인: {report.domain_pack_used}")
    print(f"claims: {len(report.claims)}, results: {len(report.results)}")
    print(f"{'='*78}\n")

    verdict_count = {}
    mismatch_count = {}
    domain_guard_count = 0
    for r in report.results:
        v = r.verdict.value
        verdict_count[v] = verdict_count.get(v, 0) + 1
        if r.mismatch_type:
            mt = r.mismatch_type.value
            mismatch_count[mt] = mismatch_count.get(mt, 0) + 1
            if mt == "domain_mismatch":
                domain_guard_count += 1

    print(f"판정 분포: {verdict_count}")
    print(f"mismatch 유형: {mismatch_count}")
    print(f"★ 도메인 가드 발동: {domain_guard_count}건\n")

    for idx, (claim, r) in enumerate(zip(report.claims, report.results), 1):
        v = r.verdict.value
        mt = r.mismatch_type.value if r.mismatch_type else "-"
        marker = " 🛡️" if mt == "domain_mismatch" else ""
        print(f"[{idx:02d}] [{v}/{mt}]{marker} conf={r.confidence:.2f}")
        if claim.schema:
            s = claim.schema
            print(f"     indicator={s.indicator!r} value={s.value} time={s.time_period!r}")
        if r.evidence:
            ev = r.evidence
            print(f"     evidence: {ev.stat_table_id} | category={ev.category_path!r}")
            print(f"               official={ev.official_value} {ev.unit}")
        print()

    out = Path("test_outputs")
    out.mkdir(exist_ok=True)
    result_json = {
        "domain": report.domain_pack_used,
        "verdict_distribution": verdict_count,
        "mismatch_distribution": mismatch_count,
        "domain_guard_triggered": domain_guard_count,
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
    json_path = out / "pipeline_temperature_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"결과 저장: {json_path.resolve()}")


asyncio.run(main())
