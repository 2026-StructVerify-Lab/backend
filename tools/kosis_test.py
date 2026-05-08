import httpx
from dotenv import load_dotenv
import os

load_dotenv()

resp = httpx.get(
    "https://kosis.kr/openapi/Param/statisticsParameterData.do",
    params={
        "method": "getList", "apiKey": os.getenv("KOSIS_API_KEY"),
        "format": "json", "jsonVD": "Y",
        "orgId": "101", "tblId": "DT_1DE9058S",
        "itmId": "ALL", "objL1": "ALL", "objL2": "ALL",
        "prdSe": "M", "startPrdDe": "200401", "endPrdDe": "200412",
    }
)
data = resp.json()
print(len(data) if isinstance(data, list) else data)