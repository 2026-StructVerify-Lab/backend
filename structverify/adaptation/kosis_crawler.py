"""
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

# KOSIS 통계표 상위 분류 코드 (주요 분야만 선별)
# 전량 수집 시에는 이 리스트를 확장하거나, 트리 탐색으로 전체 순회
KOSIS_TOP_CATEGORIES = [
    {"vw_cd": "MT_ZTITLE", "parent_id": "A"},   # 인구/가구
    {"vw_cd": "MT_ZTITLE", "parent_id": "B"},   # 고용/노동/임금
    {"vw_cd": "MT_ZTITLE", "parent_id": "F"},   # 농림수산식품
    {"vw_cd": "MT_ZTITLE", "parent_id": "H"},   # 물가/가계
    {"vw_cd": "MT_ZTITLE", "parent_id": "I"},   # 경기/기업경영
    {"vw_cd": "MT_ZTITLE", "parent_id": "N"},   # 국민계정/재정/금융
]


async def crawl_kosis_catalog(config: dict | None = None) -> list[dict[str, Any]]:
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


async def _fetch_category(
    base_url: str,
    api_key: str,
    category: dict,
    timeout: int,
) -> list[dict[str, Any]]:
    """
    단일 카테고리의 통계표 목록을 조회한다.

    TODO: httpx 실제 API 호출 구현
      GET {base_url}/Param/statisticsParameterData.do
      params:
        - apiKey: api_key
        - format: json
        - jsonList: Y
        - vwCd: category["vw_cd"]
        - parentListId: category["parent_id"]
    """
    # import httpx
    # async with httpx.AsyncClient(timeout=timeout) as client:
    #     response = await client.get(
    #         f"{base_url}/Param/statisticsParameterData.do",
    #         params={
    #             "apiKey": api_key,
    #             "format": "json",
    #             "jsonList": "Y",
    #             "vwCd": category["vw_cd"],
    #             "parentListId": category["parent_id"],
    #         },
    #     )
    #     response.raise_for_status()
    #     data = response.json()
    #     return _parse_list_response(data, category["parent_id"])

    logger.warning(f"KOSIS 카테고리 조회 stub: {category['parent_id']}")
    return []


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
    """
    수집된 메타데이터를 kosis_stat_catalog 테이블에 저장한다.
    임베딩도 함께 생성하여 embedding 컬럼에 저장 (RAG 겸용).

    TODO: asyncpg로 실제 INSERT 구현
    TODO: embedding_client.py로 임베딩 생성 → embedding 컬럼 저장

    Returns:
        int: 저장된 건수
    """
    # import asyncpg
    # conn = await asyncpg.connect(config["database"]["url"])
    # for item in catalog:
    #     embedding = await embedding_client.embed(item["stat_name"] + " " + item["description"])
    #     await conn.execute("""
    #         INSERT INTO kosis_stat_catalog
    #           (stat_id, stat_name, org_id, org_name, category_path,
    #            keywords, embedding, raw_meta_json)
    #         VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
    #         ON CONFLICT (stat_id) DO UPDATE
    #         SET stat_name=EXCLUDED.stat_name, embedding=EXCLUDED.embedding,
    #             fetched_at=NOW()
    #     """, item["stat_id"], item["stat_name"], item["org_id"],
    #          item["org_name"], item["category_path"],
    #          item["keywords"], embedding, json.dumps(item))
    # await conn.close()

    logger.warning(f"DB 저장 stub: {len(catalog)}건")
    return len(catalog)


# ── CLI 진입점 ──
if __name__ == "__main__":
    import asyncio

    async def main():
        logger.info("=== KOSIS 메타데이터 수집 시작 ===")
        catalog = await crawl_kosis_catalog()
        count = await save_to_db(catalog)
        logger.info(f"=== 완료: {count}건 저장 ===")

    asyncio.run(main())
