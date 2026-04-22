import os
import uuid
import json
import psycopg2
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI 클라이언트 ──
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Pydantic 스키마 ──
class Claim(BaseModel):
    field_name: str = Field(description="지표명 (간결하게)")
    field_value: float = Field(description="단위가 제거되고 기본 단위로 환산된 순수 숫자 (예: 3200만배럴 -> 32000000)")
    unit: str = Field(description="기본 단위 (예: 배럴, 원, 명, %)")
    is_approximate: bool = Field(description="안팎, 이상, 이하 등의 표현이 있으면 true, 명확한 수치면 false")
    modifier: Optional[str] = Field(description="근사치 수식어 (예: 안팎, 이상). 없으면 null")
    parent_path: str = Field(
        description="수치의 의미적 속성을 나타내는 대분류/중분류/소분류 계층 구조. 절대 기사 제목, 출처, 기관명을 쓰지 말 것."
    )
    time_reference: str = Field(description="기준 시점")
    context: str = Field(description="해당 수치가 등장한 원문 문장 그대로")

class ExtractResponse(BaseModel):
    claims: List[Claim]

# ── 프롬프트 ──
SYSTEM_PROMPT = """당신은 문서에서 수치가 포함된 사실 주장을 추출하는 최고 수준의 데이터 엔지니어입니다.
다음 문서에서 숫자가 포함된 모든 사실 주장을 추출하세요.

[핵심 규칙]
1. 단위 통일: '만', '억', '천' 등의 한글 단위는 숫자로 변환 (예: 21만7천명 → field_value: 217000, unit: "명")
2. 근사치 판단: 문장에 '안팎', '이상', '약' 등이 있으면 is_approximate를 true로 설정하고 modifier에 해당 단어 기록
3. 계층 분류(parent_path): 수치가 의미하는 바를 3단계 카테고리로 명확히 분류. '산업부 브리핑' 같은 출처 텍스트 사용 금지.
4. 하나의 문장에 여러 수치가 있으면 각각 별도의 객체로 분리해서 추출
5. 전수 추출 (누락 방지): 문맥의 중요도를 자체적으로 판단하지 말고, 숫자가 포함된 모든 객관적 명제를 예외 없이 전부 추출할 것.
6. 모든 항목은 반드시 모든 필드를 포함해야 함. 빠뜨리지 말 것."""


def extract_claims(text: str) -> list[dict]:
    """기사 텍스트에서 claims 추출"""
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        response_format=ExtractResponse,
        temperature=0
    )
    return response.choices[0].message.parsed.claims


def save_to_db(claims: list, source_type: str = "web_crawl", domain: str = ""):
    """추출된 claims를 DB에 적재"""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    cur = conn.cursor()

    request_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO requests (request_id, source_type, domain)
        VALUES (%s, %s, %s)
    """, (request_id, source_type, domain))

    for claim in claims:
        cur.execute("""
            INSERT INTO claims (claim_id, request_id, field_name, field_value,
                              unit, is_approximate, modifier, parent_path,
                              time_reference, context)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            str(uuid.uuid4()),
            request_id,
            claim.field_name,
            claim.field_value,
            claim.unit,
            claim.is_approximate,
            claim.modifier,
            claim.parent_path,
            claim.time_reference,
            claim.context
        ))

    conn.commit()
    cur.close()
    conn.close()
    return request_id


# ── 실행 ──
if __name__ == "__main__":
    text = """# 세대 거듭할수록 느는 \'쉬었음 청년\'…20년새 8만→22만명 | 연합뉴스\n\n세대를 거듭할수록 \'쉬었음\' 청년이 늘어나고 첫 취업까지 걸리는 기간도 길어지는 것으로 나타났다.\n연도별로 보면 쉬었음 청년은 2023년 이래 3년 연속 증가했는데 대졸 이상 고학력자가 증가세를 주도했다.\n대졸 이상 쉬었음 청년은 2023년 증가세로 돌아서며 15만3천명을 기록했고 2024년 17만4천명, 2025년 17만9천명으로 늘었다.\n전체 내용을 이해하기 위해서는 기사 본문과 함께 읽어야 합니다.\n송고 2026년04월20일 12시00분\n세대를 거듭할수록 \'쉬었음\' 청년이 늘어나고 첫 취업까지 걸리는 기간도 길어지는 것으로 나타났다.\n연도별로 보면 쉬었음 청년은 2023년 이래 3년 연속 증가했는데 대졸 이상 고학력자가 증가세를 주도했다.\n대졸 이상 쉬었음 청년은 2023년 증가세로 돌아서며 15만3천명을 기록했고 2024년 17만4천명, 2025년 17만9천명으로 늘었다.\n(서울=연합뉴스) 홍규빈 기자 = 세대를 거듭할수록 \'쉬었음\' 청년이 늘어나고 첫 취업까지 걸리는 기간도 길어지는 것으로 나타났다.\n20일 한국경영자총협회가 발표한 \'청년 일자리 창출을 위한 개선 과제\'에 따르면 2024년 1995∼1999년생의 쉬었음 인구(당시 25∼29세)는 총 21만7천명으로 집계됐다.\n이는 이전 세대와 비교해 크게 늘어난 수준이다. 2004년 당시 1975∼1979년생 쉬었음 인구(8만4천명)의 2.6배에 달한다.\n1980∼1984년생은 13만6천명(2009년 기준), 1985∼1989년생은 10만6천명(2014년), 1990∼1994년생은 16만1천명(2019년)으로 집계됐다.\n연도별로 보면 쉬었음 청년(15∼29세)은 2023년 이래 3년 연속 증가했는데 대졸 이상 고학력자가 증가세를 주도했다.\n대졸 이상 쉬었음 청년은 2023년 증가세로 돌아서며 15만3천명을 기록했고 2024년 17만4천명, 2025년 17만9천명으로 늘었다.\n고졸 이하 쉬었음 청년은 2022년 25만7천명, 2023년과 2024년 각각 24만7천명, 2025년 25만명으로 큰 변동을 보이지 않았다.\n최근에 태어난 세대일수록 첫 취업까지 걸리는 기간도 길었다.\n1995∼1999년생이 학교 졸업 후 첫 취업까지 걸린 기간은 12.77개월(2024년 기준)로, 1975∼79년생의 10.71개월(2004년)보다 2.06개월 길었다.\n1980∼1984년생은 10.70개월(2009년)이었고, 1985∼1989년생(2014년)과 1990∼1994년생(2019년)은 나란히 12.05개월이었다.\n연도별로 보면 청년층(15~29세)의 첫 취업 평균 소요 기간은 2021년 10.1개월에서 2025년 11.3개월로 늘었다.\n고졸 이하는 같은 기간 14.2개월에서 16.5개월로 늘었고 대졸 이상도 7.7개월에서 8.8개월로 길어졌다.\n한편 신규 채용으로 분류되는 \'근속 1년 미만자\' 가운데 청년층 비중은 2006년 33.6%에서 2025년 25.2%로 20년 새 8.4%포인트 하락했다.\n경총은 청년 고용 부진의 원인으로 인력수급 미스매치, 정년 60세 의무화, 저성장 고착화 등을 꼽았다.\n경총에 따르면 지난해 대기업 정규직 청년의 시간당 임금은 2만125원으로, 중소기업·비정규직 청년(1만4천66원)보다 43% 높았다.\n아울러 정년 60세가 법제화된 2013년 당시 대기업 정규직 내 청년·고령자 근로자 수를 각각 100으로 놓을 때 2025년 고령자는 245.9로 증가했고 청년은 2025년 135.5에 그쳤다.\n최문석 경총 청년ESG팀장은 "최근 청년고용률이 23개월 연속 줄어들고 20~30대 쉬었음 청년이 작년 70만명을 넘어서는 등 청년고용 위기가 이어지고 있다"며 "쉬는 청년을 노동시장으로 유인하고 일하고 싶은 청년에게 일할 기회를 제공하는 특단의 대책이 필요하다"고 강조했다.\nbingo@yna.co.kr\n제보는 카카오톡 okjebo <저작권자(c) 연합뉴스, 무단 전재-재배포, AI 학습 및 활용 금지> 2026년04월20일 12시00분 송고"""

    # 1. 추출
    claims = extract_claims(text)
    print(f"✅ {len(claims)}건 추출 완료")

    # 2. 적재
    request_id = save_to_db(claims, source_type="web_crawl", domain="경제")
    print(f"✅ DB 적재 완료 (request_id: {request_id})")

    # 3. 확인
    for c in claims:
        print(f"  - {c.field_name}: {c.field_value} {c.unit}")