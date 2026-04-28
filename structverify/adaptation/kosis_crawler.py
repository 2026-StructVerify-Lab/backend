"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-27
# 수정 내용: KOSIS 전체 통계표 메타데이터 수집 및 pgvector INSERT 구현

# [DONE] _fetch_category KOSIS API 실제 호출 구현
# [DONE] save_to_db pgvector 임베딩 INSERT 구현
# [DONE] 주제별 통계(MT_ZTITLE) 카테고리 전체 확장
# [DONE] save_to_db 배치 임베딩으로 최적화 (100건 단위)
# [DONE] NCP 임베딩 모델로 교체 (HCX 임베딩 v2, 1024차원)
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

    config = config or {}
    kosis_config = config.get("kosis", {})
    api_key = os.environ.get(kosis_config.get("api_key_env", "KOSIS_API_KEY"), "")
    base_url = kosis_config.get("base_url", "https://kosis.kr/openapi")
    timeout = kosis_config.get("timeout", 30)

    if not api_key:
        logger.error("KOSIS_API_KEY 환경변수가 설정되지 않았습니다")
        return []

    all_tables: dict[str, dict] = {}

    for category in KOSIS_TOP_CATEGORIES:
        try:
            tables = await _fetch_category(base_url, api_key, category, timeout)
            for t in tables:
                all_tables[t["stat_id"]] = t
            logger.info(f"카테고리 {category['parent_id']} 수집: {len(tables)}건")
        except Exception as e:
            logger.error(f"카테고리 {category['parent_id']} 수집 실패: {e}")

    result = list(all_tables.values())
    logger.info(f"KOSIS 메타데이터 수집 완료: 총 {len(result)}건 (중복 제거됨)")
    return result


async def _fetch_category(base_url, api_key, category, timeout):
    import httpx, re

    results = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def fetch_recursive(parent_id, path_history, depth=0):
            if depth > 7:
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


def _extract_keywords(stat_name: str) -> list[str]:
    stopwords = {"및", "의", "에", "별", "현황", "통계", "기준", "연도", "시도"}
    tokens = stat_name.replace("(", " ").replace(")", " ").split()
    return [t for t in tokens if len(t) >= 2 and t not in stopwords]


async def save_to_db(catalog: list[dict], config: dict | None = None) -> int:
    import psycopg2
    import httpx as hx
    from dotenv import load_dotenv
    load_dotenv()

    # [v1] - 박재윤: OpenAI 임베딩 (API 키 문제로 교체)
    # from openai import OpenAI
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # [v2] - 박재윤: HCX 임베딩 v2로 교체 (1024차원)
    hcx_api_key = os.environ.get("CLOVASTUDIO_API_KEY", "")

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

        # [v2] - 박재윤: HCX는 건별 호출 (배치 미지원)
        embeddings_list = []
        for item in batch:
            text = f"{item['category_path']} {item['stat_name']}"
            resp = hx.post(
                "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2",
                headers={
                    "Authorization": f"Bearer {hcx_api_key}",
                    "Content-Type": "application/json"
                },
                json={"text": text},
                timeout=30
            )
            embeddings_list.append(resp.json()["result"]["embedding"])

        for item, embedding in zip(batch, embeddings_list):
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
                str(embedding),
                json.dumps(item)
            ))

        conn.commit()
        print(f"✅ {i+len(batch)}/{len(catalog)}건 저장 완료")

    cur.close()
    conn.close()
    return len(catalog)


if __name__ == "__main__":
    import asyncio

    async def main():
        logger.info("=== KOSIS 메타데이터 수집 시작 ===")
        
        CACHE_FILE = "kosis_catalog_cache.json"
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            logger.info(f"캐시에서 로드: {len(catalog)}건")
        else:
            catalog = await crawl_kosis_catalog()
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(catalog, f, ensure_ascii=False)
            logger.info("캐시 저장 완료")

        for item in catalog[:3]:
            print(item)

        # count = await save_to_db(catalog)
        # logger.info(f"=== 완료: {count}건 저장 ===")

    asyncio.run(main())
