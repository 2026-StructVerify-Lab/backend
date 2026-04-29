import httpx
import os
from dotenv import load_dotenv
load_dotenv()

KOSIS_API_KEY = os.getenv("KOSIS_API_KEY")


resp = httpx.get(
    "https://kosis.kr/openapi/statisticsList.do",
    params={
        "method": "getList",
        "apiKey": KOSIS_API_KEY,
        "format": "json",
        "jsonVD": "Y",
        "orgId": "101",
        "tblId": "DT_1DA7147S",
    }
)
print(resp.text[:3000])