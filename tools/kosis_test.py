import httpx
import os
from dotenv import load_dotenv

load_dotenv()
KOSIS_API_KEY = os.getenv("KOSIS_API_KEY")

def extract_table_meta(org_id: str, tbl_id: str):
    base_params = {
        "method": "getList",
        "apiKey": KOSIS_API_KEY,
        "format": "json",
        "jsonVD": "Y",
        "orgId": org_id,
        "tblId": tbl_id,
        "itmId": "ALL",
        "objL1": "ALL",
        "prdSe": "Y", 
        "newEstPrdCnt": "1" # 딱 최근 1사이클만 가져와서 메타 추출용으로 사용
    }

    resp = httpx.get("https://kosis.kr/openapi/Param/statisticsParameterData.do", params=base_params, timeout=10)
    data = resp.json()

    # [재윤님이 짠 에스컬레이션 로직 적용]
    if isinstance(data, dict) and data.get("err") == "20":
        for level in range(2, 9):
            base_params[f"objL{level}"] = "ALL"
            resp = httpx.get("https://kosis.kr/openapi/Param/statisticsParameterData.do", params=base_params, timeout=10)
            data = resp.json()
            if not (isinstance(data, dict) and data.get("err") == "20"):
                break

    if isinstance(data, dict) and "err" in data:
        print(f"❌ 데이터 조회 실패: {data}")
        return None

    # 데이터 확보 성공 -> 메타데이터 파싱 및 중복 제거
    items, categories, units = set(), set(), set()

    for row in data:
        if row.get("ITM_NM"): items.add(row["ITM_NM"])
        if row.get("UNIT_NM"): units.add(row["UNIT_NM"])
        for key in row.keys():
            if "OBJ_NM" in key and row[key]:  # C1_OBJ_NM, C2_OBJ_NM 등 모두 수집
                categories.add(row[key])

    return {
        "stat_name": data[0].get("TBL_NM", ""),
        "items": ", ".join(items),
        "categories": ", ".join(categories),
        "units": ", ".join(units)
    }

if __name__ == "__main__":
    print("🔍 KOSIS 테이블 메타데이터 추출 테스트")
    meta = extract_table_meta("101", "DT_1DA7147S")
    
    if meta:
        print(f"\n📊 통계명: {meta['stat_name']}")
        print(f"🔹 포함항목: {meta['items']}")
        print(f"🔸 세부분류: {meta['categories']}")
        print(f"📏 측정단위: {meta['units']}")
        
        # 실제 임베딩에 들어갈 텍스트 조립 예시
        embedding_text = f"{meta['stat_name']} | 항목: {meta['items']} | 분류: {meta['categories']} | 단위: {meta['units']}"
        print(f"\n🧠 [최종 임베딩 텍스트]\n{embedding_text}")