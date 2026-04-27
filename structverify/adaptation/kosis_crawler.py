"""
adaptation/kosis_crawler.py — KOSIS 통계표 메타데이터 전량 수집 (Step 0-1)

[김예슬 - 2026-04-24]
- _fetch_category(): httpx 실제 HTTP 호출 구현
  · GET statisticsParameterData.do → 통계표 목록 조회
  · raise_for_status() + 응답 JSON 파싱
- _parse_list_response(): TBL_ID/TBL_NM/ORG_ID/PRD_DE 파싱 구현
  · available_periods: PRD_DE 필드 파싱 추가
  · 에러 응답(err 필드) 처리
- _extract_keywords(): 불용어 제거 + 2자 이상 필터

[참고] Self-Instruct (Wang et al., ACL 2023)
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

KOSIS_TOP_CATEGORIES = [
    {"vw_cd": "MT_ZTITLE", "parent_id": "A"},   # 인구/가구
    {"vw_cd": "MT_ZTITLE", "parent_id": "B"},   # 고용/노동/임금
    {"vw_cd": "MT_ZTITLE", "parent_id": "F"},   # 농림수산식품
    {"vw_cd": "MT_ZTITLE", "parent_id": "H"},   # 물가/가계
    {"vw_cd": "MT_ZTITLE", "parent_id": "I"},   # 경기/기업경영
    {"vw_cd": "MT_ZTITLE", "parent_id": "N"},   # 국민계정/재정/금융
]

KOSIS_LIST_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


async def crawl_kosis_catalog(config: dict | None = None) -> list[dict[str, Any]]:
    """KOSIS API → 통계표 메타데이터 전량 수집"""
    config = config or {}
    kosis_cfg = config.get("kosis", {})
    api_key = os.environ.get(kosis_cfg.get("api_key_env", "KOSIS_API_KEY"), "")
    timeout = kosis_cfg.get("timeout", 30)

    if not api_key:
        logger.error("KOSIS_API_KEY 환경변수 미설정")
        return []

    all_tables: dict[str, dict] = {}

    for category in KOSIS_TOP_CATEGORIES:
        try:
            tables = await _fetch_category(api_key, category, timeout)
            for t in tables:
                all_tables[t["stat_id"]] = t
            logger.info(f"카테고리 {category['parent_id']}: {len(tables)}건")
        except Exception as e:
            logger.error(f"카테고리 {category['parent_id']} 실패: {e}")

    result = list(all_tables.values())
    logger.info(f"KOSIS 수집 완료: {len(result)}건 (중복 제거)")
    return result


async def _fetch_category(
    api_key: str,
    category: dict,
    timeout: int,
) -> list[dict[str, Any]]:
    """단일 카테고리 통계표 목록 HTTP 조회"""
    params = {
        "apiKey":       api_key,
        "format":       "json",
        "jsonList":     "Y",
        "vwCd":         category["vw_cd"],
        "parentListId": category["parent_id"],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(KOSIS_LIST_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    return _parse_list_response(data, category["parent_id"])


def _parse_list_response(raw: Any, category_id: str) -> list[dict[str, Any]]:
    """
    KOSIS statisticsParameterData 응답 파싱.

    응답 구조:
      정상: [{TBL_ID, TBL_NM, ORG_ID, ORG_NM, STAT_NM, PRD_DE}, ...]
      에러: {"err": "30", "errMsg": "..."}
    """
    if isinstance(raw, dict) and raw.get("err"):
        logger.warning(f"KOSIS 에러 (카테고리 {category_id}): {raw.get('errMsg')}")
        return []

    items = raw if isinstance(raw, list) else raw.get("result", [])
    tables = []

    for item in items:
        stat_id = item.get("TBL_ID", "")
        if not stat_id:
            continue

        # 수집 가능 기간 파싱 ("2020,2021,2022" 형태)
        prd_de = item.get("PRD_DE", "")
        periods = [p.strip() for p in prd_de.split(",") if p.strip()] if prd_de else []

        tables.append({
            "stat_id":          stat_id,
            "stat_name":        item.get("TBL_NM", ""),
            "org_id":           item.get("ORG_ID", ""),
            "org_name":         item.get("ORG_NM", ""),
            "category_path":    item.get("STAT_NM", item.get("LIST_NM", "")),
            "keywords":         _extract_keywords(item.get("TBL_NM", "")),
            "available_periods": periods,
            "description":      item.get("TBL_NM", ""),
        })

    return tables


def _extract_keywords(stat_name: str) -> list[str]:
    stopwords = {"및", "의", "에", "별", "현황", "통계", "기준", "연도", "시도"}
    tokens = stat_name.replace("(", " ").replace(")", " ").split()
    return [t for t in tokens if len(t) >= 2 and t not in stopwords]


async def save_to_db(catalog: list[dict], config: dict | None = None) -> int:
    """
    수집된 메타데이터 → kosis_stat_catalog 테이블 저장.
    TODO [박재윤]: asyncpg INSERT 구현
    TODO [김예슬]: embedding_client로 임베딩 생성 후 저장
    """
    logger.warning(f"DB 저장 stub: {len(catalog)}건")
    return len(catalog)


if __name__ == "__main__":
    import asyncio

    async def main():
        catalog = await crawl_kosis_catalog()
        await save_to_db(catalog)
        logger.info(f"완료: {len(catalog)}건")

    asyncio.run(main())