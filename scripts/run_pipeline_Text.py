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

TEXT = """# 작년 연평균기온 14.8도…사상 처음 14도 돌파 또 신기록

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
지난여름 기온이 41도를 기록(8월 4일 경기 여주시 점동면)한 사례가 있을 정도로 최악의 폭염이 나타났다. 작년 열두 달 중 평균기온이 평년기온보다 낮은 달은 단 한 달도 없었다.
특히 늦더위가 지루하게 이어진 9월은 평균기온(25.3도)이 평년기온(20.8도)보다 4.5도나 높았다. 그나마 예년 기온을 지킨 달은 5월인데 이때도 평균기온(18.2도)이 평년기온(17.5도)을 0.7도 웃돌았다.
지난해 '덥지 않은 달'이 없었기에 기온 신기록이 수립된 것이다.
작년 더웠던 근본적인 원인으로는 기후변화를 꼽을 수밖에 없다.
우리나라 연평균 기온 순위를 보면 상위 10위 중 1998년(4위)과 1990년(9위)을 제외하고 모두 2000년 이후다.
2020년부터 작년까지 5년은 역사상 제일 뜨거웠던 5년이라고 할 수 있다.
2024년(연평균 기온이 역대 1위), 2023년(2위), 2021년(3위), 2020년(8위) 등 4개년은 열 손가락에 꼽힐 정도로 기온이 높았고 그나마 가장 '시원'했던 2022년도 연평균 기온 순위가 상위 9위이기 때문이다.
작년 우리나라뿐 아니라 지구 전체가 뜨거웠던 점도 지난해 우리가 겪은 '극한더위'의 근본 원인이 기후변화란 점을 보여준다.
유럽연합(EU)의 기후변화 감시 기구인 코페르니쿠스 기후변화연구소(C3S)에 따르면 작년 1∼11월 평균 지구 표면 기온이 1991∼2020년 평균보다 0.85도 높았다.
이는 아직 산업화 이래 가장 뜨거웠던 해인 재작년 같은 기간 온도보다 0.18도 높은 것이다.
연구소는 작년 지구 기온이 사상 처음 산업화 이전보다 1.5도 이상 높아 역대 가장 뜨거운 해가 될 것이 확실하다고 밝혔다.
'1.5도 상승'은 인류가 설정한 일종의 마지노선인데 이것이 뚫린 셈이다.
세계는 2015년 파리협정을 통해 산업화 이전 대비 지구 온도 상승 폭을 1.5도 이하로 제한하는 데 노력을 다하자고 합의했다.
과학자들은 산업화 이전 대비 지구 온도 상승 폭이 일시적으로 1.5도를 넘는 '오버슈트'만으로도 전 지구에 돌이킬 수 없는 영향이 남을 수 있다고 우려한다.
jylee24@yna.co.kr
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

    json_path = out / "pipeline_text_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n결과 저장: {json_path.resolve()}")


asyncio.run(main())