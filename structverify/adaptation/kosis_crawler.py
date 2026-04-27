"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-27
# 수정 내용: KOSIS 전체 통계표 메타데이터 수집 및 pgvector INSERT 구현

# [DONE] _fetch_category KOSIS API 실제 호출 구현
# [DONE] save_to_db pgvector 임베딩 INSERT 구현
# [DONE] 주제별 통계(MT_ZTITLE) 카테고리 전체 확장
# [DONE] save_to_db 배치 임베딩으로 최적화 (100건 단위)
# [TODO] NCP 임베딩 모델로 교체 (현재 OpenAI text-embedding-3-small 임시 사용)
# [TODO] asyncpg로 마이그레이션 (현재 psycopg2 임시 사용)
# [TODO] 기관별 통계(MT_OTITLE) 수집 추가 (별도 배치 스크립트)


adaptation/kosis_crawler.py — KOSIS 통계표 메타데이터 전량 수집 (Step 0-1)

서비스 시작 전 1회 실행하여 KOSIS의 모든 통계표 메타데이터를 수집한다.
수집된 데이터는 2가지 용도로 사용:
  1) RAG — 임베딩하여 pgvector에 저장 (방법 1)
  2) Self-Instruct — 합성 학습 데이터 생성의 seed (방법 3)

[참고] Self-Instruct (Wang et al., ACL 2023)
  - https://github.com/yizhongw/self-instruct
  - seed 데이터(통계표 메타)로부터 학습 데이터를 자동 생성하는 출발점

사용법:
  python -m adaptation.kosis_crawler          # CLI 실행
  await crawl_kosis_catalog(config)           # 코드에서 호출
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# [기존] - 박재윤: 기존 카테고리 (ID 오류)
# KOSIS_TOP_CATEGORIES = [
#     {"vw_cd": "MT_ZTITLE", "parent_id": "A"},   # 인구/가구
#     {"vw_cd": "MT_ZTITLE", "parent_id": "B"},   # 고용/노동/임금
#     {"vw_cd": "MT_ZTITLE", "parent_id": "F"},   # 농림수산식품
#     {"vw_cd": "MT_ZTITLE", "parent_id": "H"},   # 물가/가계
#     {"vw_cd": "MT_ZTITLE", "parent_id": "I"},   # 경기/기업경영
#     {"vw_cd": "MT_ZTITLE", "parent_id": "N"},   # 국민계정/재정/금융
# ]

# [v1] - 박재윤: 실제 KOSIS API 확인 후 전체 카테고리로 확장
KOSIS_TOP_CATEGORIES = [
    {"vw_cd": "MT_ZTITLE", "parent_id": "A"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "B"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "C"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "D"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "E"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "F"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "G"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "H1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "H2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "I1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "I2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "J1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "J2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "K1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "K2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "L"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "M1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "M2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "N1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "N2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "O"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "P1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "P2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "Q"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "R"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "S1"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "S2"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "T"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "U"},
    {"vw_cd": "MT_ZTITLE", "parent_id": "V"},
    # {"vw_cd": "MT_OTITLE", "parent_id": ""},   # 기관별 전체
]


async def crawl_kosis_catalog(config: dict | None = None) -> list[dict[str, Any]]:
    from dotenv import load_dotenv
    load_dotenv()
    """
    KOSIS API를 호출하여 통계표 메타데이터를 전량 수집한다.

    처리 흐름:
      1. 상위 분류별로 getList API 반복 호출
      2. 각 통계표의 메타데이터 (ID, 이름, 기관, 분류경로, 키워드) 파싱
      3. 중복 제거 후 반환

    Returns:
        list[dict]: 수집된 통계표 메타데이터 리스트
        [
          {
            "stat_id": "DT_1EA1019",
            "stat_name": "경영주 연령별 농가",
            "org_id": "101",
            "org_name": "통계청",
            "category_path": "농림수산식품 > 농림어업조사 > 농업",
            "keywords": ["경영주", "연령", "농가", "과수"],
            "available_periods": ["2020", "2021", "2022", "2023", "2024"],
            "description": "경영주 연령대별 농가 수 통계"
          },
          ...
        ]
    """
    config = config or {}
    kosis_config = config.get("kosis", {})
    api_key = os.environ.get(kosis_config.get("api_key_env", "KOSIS_API_KEY"), "")
    base_url = kosis_config.get("base_url", "https://kosis.kr/openapi")
    timeout = kosis_config.get("timeout", 30)

    if not api_key:
        logger.error("KOSIS_API_KEY 환경변수가 설정되지 않았습니다")
        return []

    all_tables: dict[str, dict] = {}  # stat_id 기준 중복 제거

    for category in KOSIS_TOP_CATEGORIES:
        try:
            tables = await _fetch_category(
                base_url, api_key, category, timeout
            )
            for t in tables:
                all_tables[t["stat_id"]] = t
            logger.info(
                f"카테고리 {category['parent_id']} 수집: {len(tables)}건"
            )
        except Exception as e:
            logger.error(f"카테고리 {category['parent_id']} 수집 실패: {e}")

    result = list(all_tables.values())
    logger.info(f"KOSIS 메타데이터 수집 완료: 총 {len(result)}건 (중복 제거됨)")
    return result


async def _fetch_category(base_url, api_key, category, timeout):
    import httpx, re, json
    
    results = []
    
    async with httpx.AsyncClient(timeout=timeout) as client:  # ← 밖으로 뺌
        async def fetch_recursive(parent_id, path_history, depth=0):
            if depth > 7:  # ← 깊이 제한
                return
                
            url = (
                f"https://kosis.kr/openapi/statisticsList.do?method=getList"
                f"&apiKey={api_key}&vwCd={category['vw_cd']}"
                f"&parentListId={parent_id}&format=json"
            )
            response = await client.get(url)
            text = response.text.strip()
            try:
                data = response.json()
            except Exception:
                fixed = re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', text)
                try:
                    data = json.loads(fixed)
                except Exception:
                    return
            
            if not isinstance(data, list):
                return
                
            for item in data:
                if 'TBL_ID' in item:
                    results.append({
                        "stat_id": item['TBL_ID'],
                        "stat_name": item.get('TBL_NM', ''),
                        "org_id": item.get('ORG_ID', ''),
                        "org_name": item.get('ORG_NM', ''),
                        "category_path": path_history,
                        "keywords": _extract_keywords(item.get('TBL_NM', '')),
                        "available_periods": [],
                        "description": item.get('TBL_NM', ''),
                    })
                elif 'LIST_ID' in item:
                    current_path = f"{path_history} > {item.get('LIST_NM', '')}"
                    await fetch_recursive(item['LIST_ID'], current_path, depth+1)
        
        await fetch_recursive(category['parent_id'], category['vw_cd'])
    
    return results


def _parse_list_response(
    raw: dict[str, Any], category_id: str
) -> list[dict[str, Any]]:
    """
    KOSIS getList API 응답을 파싱하여 통계표 메타데이터 리스트로 변환한다.

    TODO: 실제 KOSIS JSON 응답 구조에 맞게 파싱 구현
    """
    tables = []
    # KOSIS 응답 구조 예시:
    # [{"TBL_ID": "DT_1EA1019", "TBL_NM": "경영주 연령별 농가",
    #   "ORG_ID": "101", "ORG_NM": "통계청", ...}, ...]
    for item in raw if isinstance(raw, list) else raw.get("result", []):
        stat_id = item.get("TBL_ID", "")
        if not stat_id:
            continue
        tables.append({
            "stat_id": stat_id,
            "stat_name": item.get("TBL_NM", ""),
            "org_id": item.get("ORG_ID", ""),
            "org_name": item.get("ORG_NM", ""),
            "category_path": item.get("LIST_NM", ""),
            "keywords": _extract_keywords(item.get("TBL_NM", "")),
            "available_periods": [],  # TODO: 별도 API 호출로 확인
            "description": item.get("TBL_NM", ""),
        })
    return tables


def _extract_keywords(stat_name: str) -> list[str]:
    """
    통계표 이름에서 검색 키워드를 추출한다.

    TODO: 형태소 분석(Mecab/Komoran)으로 명사 추출 고도화
    현재는 공백 분리 + 불용어 제거만 수행
    """
    stopwords = {"및", "의", "에", "별", "현황", "통계", "기준", "연도", "시도"}
    tokens = stat_name.replace("(", " ").replace(")", " ").split()
    return [t for t in tokens if len(t) >= 2 and t not in stopwords]


async def save_to_db(catalog: list[dict], config: dict | None = None) -> int:
    import psycopg2
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    cur = conn.cursor()

    BATCH_SIZE = 100
    for i in range(0, len(catalog), BATCH_SIZE):
        batch = catalog[i:i+BATCH_SIZE]
        
        # 배치 임베딩
        texts = [f"{item['category_path']} {item['stat_name']}" for item in batch]
        embeddings = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        ).data
        
        for item, emb in zip(batch, embeddings):
            cur.execute("""
                INSERT INTO kosis_stat_catalog
                    (stat_id, stat_name, org_id, org_name, category_path,
                     keywords, embedding, raw_meta_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (stat_id) DO UPDATE
                SET stat_name=EXCLUDED.stat_name,
                    embedding=EXCLUDED.embedding,
                    fetched_at=NOW()
            """, (
                item["stat_id"],
                item["stat_name"],
                item["org_id"],
                item["org_name"],
                item["category_path"],
                item["keywords"],
                str(emb.embedding),
                json.dumps(item)
            ))
        
        conn.commit()
        print(f"✅ {i+len(batch)}/{len(catalog)}건 저장 완료")  # 진행상황 확인용

    cur.close()
    conn.close()
    return len(catalog)


# ── CLI 진입점 ──
if __name__ == "__main__":
    import asyncio

    async def main():
        logger.info("=== KOSIS 메타데이터 수집 시작 ===")
        catalog = await crawl_kosis_catalog()

        # 저장 전에 먼저 확인
        for item in catalog[:3]:  # 5개만
            print(item)

        count = await save_to_db(catalog)
        logger.info(f"=== 완료: {count}건 저장 ===")

    asyncio.run(main())
