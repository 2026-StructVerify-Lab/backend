import httpx
import os
from dotenv import load_dotenv
load_dotenv()

KOSIS_API_KEY = os.getenv("KOSIS_API_KEY")

resp = httpx.get(
    "https://kosis.kr/openapi/statisticsSearch.do",
    params={
        "method": "getList",
        "apiKey": KOSIS_API_KEY,
        "searchNm": "쉬었음 청년",
        "format": "json",
        "jsonVD": "Y",
        "resultCount": 5,
        "sort": "RANK",
    }
)
print(resp.status_code)
print(resp.text[:2000])