"""
★
★
★
박재윤-- 이부분은 현재 vector DB 만들어보는 TEST 공간입니다 
이부분을 토대로 
adaptation>kosis_crawler.py를 수정하고 있습니다. 
★
★
★
"""

import os
import time
import requests
import re
from dotenv import load_dotenv

load_dotenv()
KOSIS_API_KEY = os.getenv("KOSIS_API_KEY")

MAX_PRINT = 10
print_count = 0

def fetch_and_print_kosis(parent_id, path_history):
    global print_count
    
    if print_count >= MAX_PRINT:
        return

    url = (
        f"https://kosis.kr/openapi/statisticsList.do?method=getList"
        f"&apiKey={KOSIS_API_KEY}&vwCd=MT_ZTITLE"
        f"&parentListId={parent_id}&format=json"
    )
    
    try:
        response = requests.get(url, timeout=10)
        text = response.text.strip()
        
        # 🛡️ 불량 JSON 방어 로직 (방금 완벽하게 작동한 그 코드!)
        try:
            data = response.json()
        except Exception:
            fixed = re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', text)
            try:
                import json
                data = json.loads(fixed)
            except Exception:
                match = re.search(r'errMsg\s*:\s*"([^"]+)"', text)
                err_msg = match.group(1) if match else text
                print(f"    ⚠️ KOSIS API 에러 (ID: {parent_id}): {err_msg}")
                return
            
        # 정상적인 JSON일 경우 처리
        for item in data:
            if print_count >= MAX_PRINT:
                break
                
            # 1) 폴더인 경우 -> 하위 탐색 (재귀)
            if 'LIST_ID' in item and 'STAT_ID' not in item:
                current_path = f"{path_history} > {item.get('LIST_NM', '')}"
                time.sleep(0.1) # KOSIS 서버 부하 방지
                print(f"카테고리: {item.get('LIST_NM')} / ID: {item.get('LIST_ID')}")
                 # fetch_and_print_kosis(item['LIST_ID'], current_path)  # 주석처리
                
            # 2) 통계표인 경우 -> 출력
            elif 'STAT_ID' in item:
                tbl_id = item['STAT_ID']
                stat_name = item.get('TBL_NM', '알수없음')
                org_name = item.get('ORG_ID', '알수없음')
                
                print_count += 1
                print(f"[{print_count}/{MAX_PRINT}] TBL_ID: {tbl_id}")
                print(f"    👉 경로: {path_history}")
                print(f"    👉 통계명: {stat_name}")
                print(f"    👉 출처: {org_name}\n")

            

    except Exception as e:
        print(f"요청 중 네트워크 에러 발생 (ID: {parent_id}): {e}")

if __name__ == "__main__":
    print("🚀 기관별 통계목록 API 테스트를 시작합니다 (최대 10개)\n" + "="*10)
    
    # 💡 핵심 수정: "vw_cd=" 같은 군더더기 없이 순수하게 "MT_OTITLE"만 넘깁니다!
    fetch_and_print_kosis("", "주제별 통계")
    
    print("="*10)
    print("🎉 10개 테스트 출력이 완료되었습니다!")