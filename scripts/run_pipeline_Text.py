"""
scripts/run_pipeline_text.py — 멀티홉 결과 검증용 풍부한 출력판

각 claim마다 다음을 모두 출력:
  - sent_id, claim_text
  - ClaimSchema 전체 (indicator, value, unit, time_period, population)
  - 그래프에서 풀린 시점 정보 (expression, resolved, basis, via_coref)
  - Evidence (stat_table_id, official_value, unit, time_period)
  - verdict, confidence, explanation

콘솔에 표 형태로 + JSON에도 모든 필드 포함.
"""
import asyncio
import json
from pathlib import Path

from structverify.core.pipeline import VerificationPipeline
from structverify.graph.claim_graph import ClaimGraph
import random
TEXT = """# 작년 연평균기온 14.8도…사상 처음 14도 돌파 또 신기록
입력 2025.01.01 14:59
(서울=연합뉴스) 이재영 기자 = 2024년은 우리나라 연평균 기온이 사상 처음 14도를 돌파해 '압도적으로 뜨거웠던 해'로 남았다.
1일 기상청 기상자료개방포털을 보면 작년 평균기온은 14.8도로 재작년(13.9도)에 이어 2년 연속 '1973년 이후 연평균 기온 신기록'을 갈아치웠다.
1973년은 기상관측망이 대폭 확충돼 각종 기상기록 기준점이 되는 해다.
한 해 평균기온이 14도를 넘기는 작년이 처음이다.
작년 평균기온은 평년(1991∼2020년 평균) 연평균 기온(12.7±0.2도)을 2.3도나 웃돌았다.
(서울=연합뉴스) 김토일 기자 kmtoil@yna.co.kr
페이스북 tuney.kr/LeYN1 X(트위터) @yonhap_graphics
일최저기온과 일최고기온 연평균 값도 지난해가 역대 1위다.
작년 평균 최저기온은 10.2도로 10도를 넘었고 평균 최고기온은 20.1도로 20도를 웃돌았다. 최저기온과 최고기온도 평균기온과 마찬가지로 재작년에 연이어 신기록을 경신했다.
기온 기록은 기상청이 관측값 재검증을 거쳐 공식 발표할 때 달라질 수 있다.
"""


def _short(s, n=80):
    if s is None:
        return "None"
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


async def main():
    pipeline = VerificationPipeline()
    report = await pipeline.run(TEXT, "text")

    # 멀티홉 traversal용 그래프 facade 재구성
    graph = ClaimGraph(report.graph_nodes, report.graph_edges)
    anchor_year = graph.get_anchor_year()

    print(f"\n{'='*78}")
    print(f"도메인: {report.domain_pack_used}")
    print(f"anchor_year (그래프): {anchor_year}")
    print(f"graph: {len(report.graph_nodes)} nodes, {len(report.graph_edges)} edges")
    print(f"claims: {len(report.claims)}, results: {len(report.results)}")
    print(f"{'='*78}\n")

    verdict_count = {}

    for idx, (claim, r) in enumerate(zip(report.claims, report.results), 1):
        v = r.verdict.value
        verdict_count[v] = verdict_count.get(v, 0) + 1

        # 그래프에서 멀티홉 시점 해소
        prov = graph.temporal_provenance(claim)

        print(f"[{idx:02d}] {'─'*70}")
        print(f"  verdict   : [{v}]  conf={r.confidence:.2f}")
        print(f"  sent_id   : {claim.sent_id}")
        print(f"  claim     : {_short(claim.claim_text, 100)}")

        if claim.schema:
            s = claim.schema
            print(f"  schema    :")
            print(f"      indicator    = {s.indicator}")
            print(f"      value        = {s.value}  unit={s.unit!r}")
            print(f"      time_period  = {s.time_period!r}")
            print(f"      population   = {s.population!r}")
        else:
            print(f"  schema    : None")

        # 그래프 멀티홉 결과
        if prov:
            print(f"  graph 시점 해소:")
            print(f"      표현      = {prov.get('expression')!r}")
            print(f"      resolved  = {prov.get('resolved')!r}")
            print(f"      basis     = {_short(prov.get('basis'), 70)}")
            if prov.get('via_coref'):
                print(f"      coref →   {prov.get('via_coref')}")
        else:
            print(f"  graph 시점 해소: (해당 문장에 추출된 시간 표현 없음)")

        # Evidence
        if r.evidence:
            ev = r.evidence
            print(f"  evidence  :")
            print(f"      stat_id     = {ev.stat_table_id}")
            print(f"      stat_name   = {ev.source_name}")
            print(f"      official    = {ev.official_value} {ev.unit or ''}")
            print(f"      time_period = {ev.time_period!r}")
        else:
            print(f"  evidence  : None")

        if r.explanation:
            print(f"  explain   : {_short(r.explanation, 200)}")
        print()
        # 기존 schema 출력 블록 아래에 추가
        print(f"  evidence_plan :")
        plan = claim.schema.evidence_plan if claim.schema else None
        if plan and plan.requirements:
            print(f"      combiner   = {plan.combiner}")
            for req in plan.requirements:
                print(f"      [{req.role}] time={req.time_period} "
                    f"indicator={req.indicator} label={req.label}")
        else:
            print(f"      (plan 없음 또는 requirements 비어있음)")

        # value_role도
        if claim.schema:
            print(f"      value_role = {claim.schema.value_role.value}")

    print(f"{'='*78}")
    print(f"판정 분포: {verdict_count}")
    print(f"{'='*78}")
    

    # ─────────────────────────────────────────────────────
    # JSON 저장 (모든 필드 포함)
    # ─────────────────────────────────────────────────────
    out = Path("test_outputs")
    out.mkdir(exist_ok=True)

    result_json = {
        "domain": report.domain_pack_used,
        "anchor_year": anchor_year,
        "graph_stats": graph.stats(),
        "verdict_distribution": verdict_count,
        "results": [],
    }

    for c, r in zip(report.claims, report.results):
        prov = graph.temporal_provenance(c)
        ev = r.evidence

        item = {
            "sent_id": c.sent_id,
            "claim_text": c.claim_text,
            "schema": {
                "indicator":   c.schema.indicator if c.schema else None,
                "value":       c.schema.value if c.schema else None,
                "unit":        c.schema.unit if c.schema else None,
                "time_period": c.schema.time_period if c.schema else None,
                "population":  c.schema.population if c.schema else None,
                "source_reference": (c.schema.source_reference if c.schema else None),
            },
            "graph_temporal": {
                "expression": prov.get("expression"),
                "resolved":   prov.get("resolved"),
                "basis":      prov.get("basis"),
                "via_coref":  prov.get("via_coref"),
            } if prov else None,
            "evidence": {
                "stat_table_id":  ev.stat_table_id,
                "source_name":    ev.source_name,
                "official_value": ev.official_value,
                "unit":           ev.unit,
                "time_period":    ev.time_period,
            } if ev else None,
            "verdict":     r.verdict.value,
            "confidence":  r.confidence,
            "mismatch_type": r.mismatch_type.value if r.mismatch_type else None,
            "explanation": r.explanation,
        }
        result_json["results"].append(item)
    uuid = random.randint(1000, 9999)
    json_path = out / f"pipeline_text_result_{uuid}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n결과 저장: {json_path.resolve()}")


asyncio.run(main())