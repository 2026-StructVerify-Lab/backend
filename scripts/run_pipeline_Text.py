# scripts/run_pipeline_text.py
import asyncio
import json
from pathlib import Path
from structverify.core.pipeline import VerificationPipeline

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


async def main():
    pipeline = VerificationPipeline()

    report = await pipeline.run(TEXT, "text")

    print(f"\n도메인: {report.domain_pack_used}")
    print(f"claims: {len(report.claims)}, results: {len(report.results)}")

    verdict_count = {}
    for claim, r in zip(report.claims, report.results):
        v = r.verdict.value
        verdict_count[v] = verdict_count.get(v, 0) + 1
        print(f"\n  [{v}] {claim.claim_text[:60]}")
        if r.explanation:
            print(f"       {r.explanation[:80]}")

    print(f"\n판정 분포: {verdict_count}")

    out = Path("test_outputs")
    out.mkdir(exist_ok=True)

    result_json = {
        "domain": report.domain_pack_used,
        "verdict_distribution": verdict_count,
        "results": [
            {
                "claim_text":  c.claim_text,
                "indicator":   c.schema.indicator if c.schema else None,
                "value":       c.schema.value if c.schema else None,
                "verdict":     r.verdict.value,
                "confidence":  r.confidence,
                "explanation": r.explanation,
            }
            for c, r in zip(report.claims, report.results)
        ],
    }

    json_path = out / "pipeline_text_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"결과 저장: {json_path.resolve()}")


asyncio.run(main())