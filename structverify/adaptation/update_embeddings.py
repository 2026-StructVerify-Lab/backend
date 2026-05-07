"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-30
# 수정 내용: KOSIS 딥 메타데이터 수집 및 벡터 DB 재구축

# [DONE] KOSIS 메타 API로 항목명/분류명/단위 수집 후 임베딩 텍스트 보강
# [DONE] 지자체/지역 통계 제외 (197,956건만 처리)
# [DONE] 세마포어 기반 병렬 처리 (API=5, HCX=3)
# [DONE] prdSe Y→M→Q 순회 + err:20 objL 에스컬레이션
# [DONE] 429/타임아웃 재시도 로직 (최대 5회)
# [DONE] 배치 단위 DB UPDATE (ON CONFLICT 없이 순수 UPDATE)
# [TODO] 실패 건(None 반환) 별도 재시도 스크립트 작성
# [TODO] err:21 테이블 대상 별도 파라미터 방식 시도
# [TODO] 완료 후 factcheck_test.py 재실행 및 정확도 검증
"""

import os
import json
import asyncio
import httpx
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

HCX_API_KEY = os.getenv("CLOVASTUDIO_API_KEY")
KOSIS_API_KEY = os.getenv("KOSIS_API_KEY")
PG_CONN = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}

CATALOG_JSON_FILE = "kosis_catalog_cache.json"

async def fetch_kosis_deep_meta(client, org_id, tbl_id):
    """KOSIS API 상세 정보 조회 (Y->M->Q 순회 및 에러 핸들링 강화)"""
    
    # 연간(Y), 월간(M), 분기(Q) 순으로 시도
    for prdSe in ["Y", "M", "Q"]:
        base_params = {
            "method": "getList", "apiKey": KOSIS_API_KEY, "format": "json", "jsonVD": "Y",
            "orgId": org_id, "tblId": tbl_id, "itmId": "ALL", "objL1": "ALL",
            "prdSe": prdSe, "newEstPrdCnt": "1"
        }

        try:
            resp = await client.get("https://kosis.kr/openapi/Param/statisticsParameterData.do", params=base_params, timeout=15)
            data = resp.json()

            # 에러 20 (세부항목 누락) 발생 시 파라미터 에스컬레이션
            if isinstance(data, dict) and data.get("err") == "20":
                for level in range(2, 9):
                    base_params[f"objL{level}"] = "ALL"
                    resp = await client.get("https://kosis.kr/openapi/Param/statisticsParameterData.do", params=base_params, timeout=15)
                    data = resp.json()
                    if not (isinstance(data, dict) and data.get("err") == "20"):
                        break

            # 에러 30 (데이터 없음) -> 다음 시점(M, Q)으로 재시도
            if isinstance(data, dict) and data.get("err") == "30":
                continue 

            # 에러 31 (너무 큼) 등 기타 에러 -> 과감히 포기 (기본 이름으로 임베딩)
            if isinstance(data, dict) and "err" in data:
                # 에러 로그가 너무 많이 뜨면 불편하니 31번은 조용히 넘깁니다.
                if data.get("err") != "31": 
                    print(f"⚠️ KOSIS API 에러 [{tbl_id}]: {data.get('err')} - {data.get('errMsg')}")
                return None

            # 정상 데이터 파싱
            items, categories, units = set(), set(), set()
            for row in data:
                if row.get("ITM_NM"): items.add(row["ITM_NM"])
                if row.get("UNIT_NM"): units.add(row["UNIT_NM"])
                for key in row.keys():
                    if "OBJ_NM" in key and row[key]:
                        categories.add(row[key])

            return {
                "items": ", ".join(items)[:200],
                "categories": ", ".join(categories)[:300],
                "units": ", ".join(units)[:50]
            }

        except Exception as e:
            # 타임아웃 등 통신 오류 시 조용히 넘김
            return None

    # Y, M, Q 다 돌았는데도 없으면 None 반환
    return None

async def process_single_table(client, item, semaphore_api, semaphore_hcx):
    async with semaphore_api:
        meta = await fetch_kosis_deep_meta(client, item["org_id"], item["stat_id"])

    embed_text = f"{item['category_path']} > {item['stat_name']}"
    if meta:
        embed_text += f" | 항목: {meta['items']} | 분류: {meta['categories']} | 단위: {meta['units']}"

    async with semaphore_hcx:
        for retry in range(5):  # 3 → 5
            try:
                resp = await client.post(
                    "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2",
                    headers={"Authorization": f"Bearer {HCX_API_KEY}", "Content-Type": "application/json"},
                    json={"text": embed_text}, timeout=30
                )
                data = resp.json()
                if data.get("result"):
                    return (data["result"]["embedding"], item["stat_id"])
                if resp.status_code == 429:  # ← 추가
                    wait = 2 ** retry
                    await asyncio.sleep(wait)
                    continue
            except Exception:
                await asyncio.sleep(2 ** retry)
    return None

async def main():
    print("🚀 KOSIS 딥 메타데이터 수집 및 벡터 DB 재구축 시작 (안정화 버전)")

    with open(CATALOG_JSON_FILE, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    # 타겟 필터링 유지
    target_catalog = [
        item for item in catalog
        if "지자체" not in item.get("category_path", "") and "지역" not in item.get("category_path", "")
    ]

    # ⭐ 핵심 수정: 동시 접속 수 대폭 축소 (서버 부하 방지)
    semaphore_api = asyncio.Semaphore(5)   # 기존 15 -> 5 로 축소 (KOSIS 서버 과부하 방지)
    semaphore_hcx = asyncio.Semaphore(3)   # 기존 5 -> 3 로 축소 (안정적인 임베딩 요청)

    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()

    BATCH_SIZE = 100
    async with httpx.AsyncClient(timeout=30) as client:
        # 이전에 실패했던 지점(예: 600번)부터 다시 시작할 수 있도록 인덱스를 직접 설정할 수 있습니다.
        # 처음부터 다시 하려면 0으로 두세요.
        start_index = 0

        for i in range(start_index, len(target_catalog), BATCH_SIZE):
            batch = target_catalog[i:i+BATCH_SIZE]

            tasks = [process_single_table(client, item, semaphore_api, semaphore_hcx) for item in batch]
            results = await asyncio.gather(*tasks)

            update_records = [res for res in results if res is not None]

            if update_records:
                try:
                    execute_values(
                        cur,
                        "UPDATE kosis_stat_catalog SET embedding = data.vector::vector FROM (VALUES %s) AS data(vector, stat_id) WHERE kosis_stat_catalog.stat_id = data.stat_id",
                        update_records
                    )
                    conn.commit()
                except Exception as e:
                    print(f"❌ DB 업데이트 실패 (배치 {i}~{i+BATCH_SIZE}): {e}")
                    conn.rollback() # DB 업데이트 실패 시 롤백

            print(f"🔄 처리 중: {i + len(batch)} / {len(target_catalog)} 건 (성공: {len(update_records)}건)")
            await asyncio.sleep(1)  # ← 추가

    cur.close()
    conn.close()
    print("🎉 완료되었습니다!")

if __name__ == "__main__":
    asyncio.run(main())