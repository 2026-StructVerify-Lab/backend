"""
retrieval/catalog_search.py — kosis_stat_catalog pgvector 검색 모듈 (Step 7-0)

[박재윤 - 2026-04-30]
- kosis_stat_catalog 테이블 pgvector 검색 구현 (factcheck_test.py 참고)
  · search_pgvector(): category_path ILIKE 필터 + embedding 유사도 검색
  · get_embedding(): HCX 임베딩 API 호출

[김예슬 - 2026-04-30]
- extract_category_and_keyword(): LLM이 indicator → KOSIS 카테고리 + 검색어 추출
- CatalogSearchTool: KOSISConnector가 호출하는 Tool 인터페이스
  · search(): ConnectorQuery → 후보 StatRecord 목록 반환
  · 내부: keyword 검색(KOSIS API vwCd=MT_ZTITLE) + pgvector 필터 검색 + pgvector 전체 검색 조합

[설계]
  kosis_stat_catalog는 공식 데이터 저장소가 아니라
  "어떤 stat_id를 써야 하는지 찾는 검색 인덱스"로 사용.

  검색 3단계:
    1) KOSIS 통합검색(vwCd=MT_ZTITLE): 국가통계만 + RANK 순
    2) pgvector (category_path 필터 + embedding): 카테고리 범위 좁힌 유사도 검색
    3) pgvector (필터 없이): 전체 유사도 검색 (폴백)

  중복 제거 후 최대 top_k 반환 → LLM Agent가 최적 stat_id 선택
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from structverify.retrieval.base_connector import ConnectorQuery, StatRecord
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# KOSIS 통합검색에서 국가통계만 (지역통계 제외)
_KOSIS_SEARCH_VW_CD = "MT_ZTITLE"

# LLM 카테고리/검색어 추출 프롬프트
_CATEGORY_EXTRACT_PROMPT = """다음 뉴스 수치 주장을 분석하여 두 가지를 추출하세요.

indicator: {indicator}
population: {population}
원문: {claim_text}

1) 이 통계가 속할 KOSIS 카테고리 경로 키워드 2~3개 (쉼표 구분)
   KOSIS 주요 분야: 인구, 가구, 고용, 노동, 임금, 물가, 가계, 보건, 사회, 복지, 교육, 환경, 농림, 수산, 건설, 주택, 토지, 교통, 정보통신, 경제, 산업, 무역
   이 수치가 나올 법한 통계조사명도 포함하세요.

2) KOSIS 통계표 이름에 들어갈 법한 검색어 2~3단어 (숫자/연도 금지)

형식:
카테고리: 키워드1, 키워드2, 키워드3
검색어: 검색 키워드"""


class CatalogSearchTool:
    """
    kosis_stat_catalog pgvector DB를 KOSIS API 검색 인덱스로 사용하는 Tool.

    KOSISConnector.search_and_fetch()에서 1단계로 호출됨.
    """

    def __init__(self, config: dict | None = None):
        self.config  = config or {}
        self.api_key = os.environ.get(
            self.config.get("api_key_env", "KOSIS_API_KEY"), ""
        )
        self.hcx_key = os.environ.get(
            self.config.get("llm", {}).get("api_key_env", "CLOVASTUDIO_API_KEY"), ""
        )
        self.pg_dsn  = os.environ.get(
            self.config.get("pgvector_dsn_env", "PGVECTOR_DSN"),
            "postgresql://structverify:svpass123@localhost:5432/structverify",
        )
        self.timeout = self.config.get("timeout", 30)

    async def search(
        self,
        query: ConnectorQuery,
        top_k: int = 10,
    ) -> list[StatRecord]:
        """
        ConnectorQuery → 후보 StatRecord 목록.

        검색 3단계:
          1) KOSIS 통합검색 (vwCd=MT_ZTITLE, 국가통계만)
          2) pgvector category_path 필터 + embedding 검색
          3) pgvector 전체 embedding 검색 (폴백)

        중복 제거 후 top_k 반환.
        """
        # LLM으로 category 키워드 + 검색어 추출
        category_kws, search_kw = await self._extract_category_and_keyword(query)
        logger.info(f"CatalogSearch: keyword='{search_kw}' category={category_kws}")

        results: list[StatRecord] = []
        seen_ids: set[str] = set()

        def _add(recs: list[StatRecord]) -> None:
            for r in recs:
                if r.stat_id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.stat_id)

        # 1) KOSIS 통합검색
        kosis_recs = await self._search_kosis_api(search_kw, max_results=top_k)
        _add(kosis_recs)

        # 2+3) pgvector 검색
        embedding_text = (query.extra_params or {}).get("embedding_text") or (
            " ".join(filter(None, [query.indicator, query.population, search_kw]))
        )
        embedding = await self._get_embedding(embedding_text)

        if embedding:
            # 2) category 필터 + embedding
            if category_kws:
                cat_recs = await self._search_pgvector(
                    embedding,
                    category_keywords=category_kws,
                    top_k=top_k,
                )
                _add(cat_recs)

            # 3) 전체 embedding (폴백)
            all_recs = await self._search_pgvector(embedding, category_keywords=None, top_k=5)
            _add(all_recs)

        logger.info(f"CatalogSearch 완료: {len(results)}개 후보")
        return results[:top_k]

    # ── LLM 카테고리/검색어 추출 ────────────────────────────────────────────

    async def _extract_category_and_keyword(
        self, query: ConnectorQuery
    ) -> tuple[list[str], str]:
        """LLM이 indicator → KOSIS category 키워드 + 검색어 추출"""
        if not self.hcx_key:
            return ([], query.indicator or query.keyword or "")

        raw_claim = (query.extra_params or {}).get("raw_claim", "")
        prompt = _CATEGORY_EXTRACT_PROMPT.format(
            indicator=query.indicator or "",
            population=query.population or "",
            claim_text=raw_claim[:200] or query.keyword,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002",
                    headers={
                        "Authorization": f"Bearer {self.hcx_key}",
                        "Content-Type":  "application/json",
                    },
                    json={
                        "messages":    [{"role": "user", "content": prompt}],
                        "maxTokens":   80,
                        "temperature": 0,
                    },
                )
                content = resp.json()["result"]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"카테고리 추출 실패: {e}")
            return ([], query.indicator or query.keyword or "")

        category_keywords: list[str] = []
        search_keyword = query.indicator or query.keyword or ""

        for line in content.split("\n"):
            line = line.strip()
            if "카테고리" in line and ":" in line:
                cats = line.split(":", 1)[1].strip()
                category_keywords = [
                    c.strip().strip("\"'") for c in cats.split(",") if c.strip()
                ]
            elif "검색어" in line and ":" in line:
                kw = line.split(":", 1)[1].strip().strip("\"'")
                kw = re.sub(r"\b\d{4}\b", "", kw).strip()
                kw = re.sub(r"\s+", " ", kw)
                if kw:
                    search_keyword = kw

        return (category_keywords, search_keyword)

    # ── HCX 임베딩 생성 ──────────────────────────────────────────────────────

    async def _get_embedding(self, text: str) -> list[float] | None:
        """HCX 임베딩 API로 벡터 생성"""
        if not self.hcx_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2",
                    headers={
                        "Authorization": f"Bearer {self.hcx_key}",
                        "Content-Type":  "application/json",
                    },
                    json={"text": text},
                )
                return resp.json()["result"]["embedding"]
        except Exception as e:
            logger.debug(f"임베딩 생성 실패: {e}")
            return None

    # ── KOSIS 통합검색 ────────────────────────────────────────────────────────

    async def _search_kosis_api(
        self, keyword: str, max_results: int = 5
    ) -> list[StatRecord]:
        """KOSIS statisticsSearch.do — vwCd=MT_ZTITLE (국가통계만)"""
        if not self.api_key or not keyword.strip():
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    "https://kosis.kr/openapi/statisticsSearch.do",
                    params={
                        "method":       "getList",
                        "apiKey":       self.api_key,
                        "searchNm":     keyword,
                        "format":       "json",
                        "jsonVD":       "Y",
                        "resultCount":  max_results,
                        "sort":         "RANK",
                        "vwCd":         _KOSIS_SEARCH_VW_CD,  # 국가통계만
                    },
                )
                data = resp.json()
        except Exception as e:
            logger.debug(f"KOSIS 통합검색 실패: {e}")
            return []

        if isinstance(data, dict) and ("err" in data or "errMsg" in data):
            return []
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) and "TBL_ID" in data else []

        records = []
        n = len(data)
        for i, item in enumerate(data):
            tid = (item.get("TBL_ID") or "").strip()
            if not tid or not item.get("ORG_ID"):
                continue
            rel = 1.0 if n <= 1 else max(0.05, 1.0 - (i / (n - 1)) * 0.95)
            records.append(StatRecord(
                stat_id=tid,
                stat_name=item.get("TBL_NM", ""),
                org_id=item.get("ORG_ID"),
                org_name=item.get("ORG_NM"),
                relevance_score=rel,
                metadata={"source": "kosis_api", **item},
            ))

        logger.debug(f"KOSIS API 검색: {len(records)}개")
        return records

    # ── pgvector 검색 ─────────────────────────────────────────────────────────

    async def _search_pgvector(
        self,
        embedding: list[float],
        category_keywords: list[str] | None = None,
        top_k: int = 5,
    ) -> list[StatRecord]:
        """
        kosis_stat_catalog pgvector 유사도 검색.

        category_keywords 있으면: category_path ILIKE 필터 + embedding 정렬
        없으면: 전체 embedding 정렬
        """
        try:
            import asyncpg
        except ImportError:
            logger.debug("asyncpg 미설치 → pgvector 검색 skip")
            return []

        vector_str = "[" + ",".join(str(v) for v in embedding) + "]"

        try:
            conn = await asyncpg.connect(self.pg_dsn)
        except Exception as e:
            logger.debug(f"pgvector 연결 실패: {e}")
            return []

        try:
            if category_keywords:
                # category_path ILIKE 필터 + embedding 거리 정렬
                where_parts = [f"category_path ILIKE ${i+2}" for i in range(len(category_keywords))]
                where_sql   = " OR ".join(where_parts)
                params      = [vector_str] + [f"%{kw}%" for kw in category_keywords] + [top_k]
                sql = f"""
                    SELECT stat_id, stat_name, org_id, org_name, category_path, keywords,
                           1 - (embedding <-> $1::vector) AS similarity
                    FROM kosis_stat_catalog
                    WHERE ({where_sql})
                      AND embedding IS NOT NULL
                    ORDER BY embedding <-> $1::vector
                    LIMIT ${len(params)}
                """
            else:
                params = [vector_str, top_k]
                sql = """
                    SELECT stat_id, stat_name, org_id, org_name, category_path, keywords,
                           1 - (embedding <-> $1::vector) AS similarity
                    FROM kosis_stat_catalog
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> $1::vector
                    LIMIT $2
                """

            rows = await conn.fetch(sql, *params)
            await conn.close()

            records = []
            for row in rows:
                # periods = row["available_periods"] or []
                # if isinstance(periods, str):
                #     try:
                #         periods = json.loads(periods)
                #     except Exception:
                #         periods = []
                sim = float(row.get("similarity") or 0.0)
                records.append(StatRecord(
                    stat_id=row["stat_id"],
                    stat_name=row["stat_name"],
                    org_id=row["org_id"],
                    org_name=row["org_name"],
                    available_periods=[],
                    relevance_score=max(0.0, sim),
                    metadata={
                        "source":        "pgvector",
                        "category_path": row.get("category_path"),
                        "keywords":      row.get("keywords"),
                        "similarity":    sim,
                    },
                ))

            label = f"(category={category_keywords})" if category_keywords else "(전체)"
            logger.debug(f"pgvector 검색 {label}: {len(records)}개")
            return records

        except Exception as e:
            logger.warning(f"pgvector 검색 실패: {e}")
            try:
                await conn.close()
            except Exception:
                pass
            return []