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

# LLM 카테고리/검색어 추출 프롬프트 (parent_path 없을 때만 fallback으로 사용)
_CATEGORY_EXTRACT_PROMPT = """다음 뉴스 수치 주장을 KOSIS 검색에 적합한 형태로 분석하세요.

indicator: {indicator}
population: {population}
원문: {claim_text}

[추출 항목]
1) 카테고리 키워드: 이 통계가 속할 KOSIS 분야 키워드 1~3개 (쉼표 구분)
   KOSIS 대분류: 인구, 가구, 고용, 노동, 임금, 물가, 가계, 보건, 사회, 복지,
                교육, 환경, 농림, 수산, 건설, 주택, 토지, 교통, 정보통신, 경제, 산업, 무역

2) 검색 키워드: KOSIS 통계표 이름에 들어갈 *핵심 명사* 2~3 단어
   - 숫자/연도/월/일 절대 포함 금지
   - "증가율/변화/차이/상승/하락" 같은 측정 행위 단어 *제외*
   - "출생아 수" "혼인 건수" "쉬었음 인구" 같이 측정 대상 자체만

[좋은 예]
indicator="출생아 수 증가율" → 검색어="출생아 수"
indicator="혼인 건수 증가율" → 검색어="혼인 건수"
indicator="연평균 기온" → 검색어="연평균 기온"
indicator="쉬었음 청년" → 검색어="쉬었음 인구"

[형식 — 이 두 줄만 출력]
카테고리: 키워드1, 키워드2
검색어: 핵심 명사 두세개"""


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

    # ── 카테고리/검색어 추출 ────────────────────────────────────────────

    async def _extract_category_and_keyword(
        self, query: ConnectorQuery
    ) -> tuple[list[str], str]:
        """
        ConnectorQuery → (category_keywords, search_keyword) 추출.

        [v6.11] schema에 parent_path가 있으면 *LLM 호출 없이* 그대로 분해.
        없을 때만 LLM 호출 (박재유 방식의 fallback).

          · 4자리 연도 제거
          · 공백 정리
        """
        # 1) schema의 parent_path가 있으면 우선 사용 (LLM 호출 절약)
        parent_path = (query.extra_params or {}).get("parent_path")
        if parent_path:
            # "노동 > 청년 > 쉬었음 인구" → category=["노동", "청년"], keyword="쉬었음 인구"
            parts = [p.strip() for p in re.split(r"\s*>\s*", parent_path) if p.strip()]
            if parts:
                # 마지막 노드(소분류)를 검색어로
                search_kw = parts[-1]
                # 앞 1-2개 노드를 카테고리로
                category_kws = parts[:-1] if len(parts) > 1 else parts
                # 박재유 정제 (연도 + 공백)
                search_kw = _minimal_clean(search_kw)
                if search_kw:
                    logger.debug(f"parent_path 활용: category={category_kws}, kw={search_kw}")
                    return (category_kws, search_kw)

        # 2) parent_path 없으면 LLM 호출 (fallback)
        if not self.hcx_key:
            return ([], _minimal_clean(query.indicator or query.keyword or ""))

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
            return ([], _minimal_clean(query.indicator or query.keyword or ""))

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
                kw = _minimal_clean(kw)
                if kw:
                    search_keyword = kw

        # fallback도 최소 정제
        search_keyword = _minimal_clean(search_keyword)

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


# 룰베이스 stopword/action-word 매핑은 *제거*함 (사용자 원칙: LLM이 정제 책임).
#     kw = re.sub(r'\b\d{4}\b', '', kw).strip()
#     kw = re.sub(r'\s+', ' ', kw)

def _minimal_clean(kw: str) -> str:
    """LLM 응답에서 명백한 노이즈만 제거 (연도 + 공백). 의미 정제는 LLM 책임."""
    if not kw:
        return ""
    s = re.sub(r"\b\d{4}\b", "", kw).strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(",·\"' ")