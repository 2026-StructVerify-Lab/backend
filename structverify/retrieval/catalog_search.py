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


# [v6.2] LLM이 카테고리/검색어에 마크다운 마커를 섞어 반환하는 케이스 정제
# 예: "**최고기온 기록**, **" → "최고기온 기록"
_MARKDOWN_MARKER_RE = re.compile(r"[*_`#~]+")


def _strip_markdown(text: str) -> str:
    """LLM 응답에서 마크다운 마커 제거 + 양끝 따옴표/공백 정리."""
    if not text:
        return ""
    cleaned = _MARKDOWN_MARKER_RE.sub("", text)
    cleaned = cleaned.strip().strip("\"'").strip()
    return cleaned

# LLM 카테고리/검색어 추출 프롬프트
_CATEGORY_EXTRACT_PROMPT = """다음 뉴스 수치 주장을 분석하여 KOSIS 검색용 정보를 추출하세요.

indicator: {indicator}
population: {population}
원문: {claim_text}

1) KOSIS 카테고리 키워드 (쉼표 구분, 2~3개)
   ⚠️ 반드시 아래 KOSIS 공식 분야 명칭 *그대로* 사용. 자유롭게 풀어쓰지 마세요.

   [KOSIS 공식 분야 목록 — 이 단어들 중에서만 선택]
   인구, 가구, 출생, 사망, 혼인, 이혼, 가족, 고령자,
   고용, 노동, 임금, 일자리, 실업, 청년,
   물가, 가계, 소득, 소비, 자산, 부채,
   보건, 의료, 질병, 사망원인,
   사회, 복지, 연금, 보육, 장애인,
   교육, 학교, 대학, 학력,
   환경, 기상, 기후, 날씨, 대기, 수질, 폐기물, 재해,
   농림, 농업, 임업, 수산, 어업, 축산,
   건설, 주택, 토지, 부동산, 임대,
   교통, 자동차, 도로, 철도, 항공, 해운,
   정보통신, 인터넷, 통신, 방송, ICT,
   경제, GDP, 산업, 제조업, 서비스업, 무역, 수출입, 금융, 은행, 증권,
   에너지, 전력, 가스, 석유, 재생에너지

   [잘못된 예시 — 절대 이렇게 답하지 마세요]
   "기후 변화", "기온 변화", "날씨 데이터" ← 자유 풀어쓰기 (금지)
   "출생아수 통계", "혼인수 동향" ← 위 목록에 없는 표현 (금지)

   [올바른 예시]
   기온 14.8도 claim → 카테고리: 환경, 기상, 기후
   출생아수 6.7% 증가 claim → 카테고리: 인구, 출생
   혼인 건수 3.9% 증가 claim → 카테고리: 인구, 혼인
   합계출산율 0.04 증가 claim → 카테고리: 인구, 출생

2) KOSIS 통계표명에 들어갈 핵심 단어 2~3개 (쉼표 구분, 숫자/연도 금지)
   stat_name에 실제로 등장하는 **공식 KOSIS 용어**. 형용사/수식어 빼고 핵심 명사만.

   ⚠️ population (분류 축 — 대졸이상, 청년, 여성, 외국인 등)은 검색어에 넣지 마세요.
      stat_name은 *지표 본체*로 매칭. 분류 축은 표 내부 분해 컬럼에 있음.

   [KOSIS 자주 등장 공식 용어 (이걸 우선 사용)]
   고용/노동: 쉬었음, 비경제활동인구, 경제활동인구, 취업자수, 실업률, 임금근로자, 일자리
   인구: 출생아수, 사망자수, 혼인건수, 이혼건수, 합계출산율, 추계인구, 1인가구
   가구/소득: 가구소득, 가처분소득, 경상소득, 가계지출, 소비지출
   환경/기상: 평균기온, 최저기온, 최고기온, 강수량, 일조시간, 미세먼지, PM10, PM2.5
   교육: 진학률, 졸업자수, 재학생수, 사교육비

   [올바른 예시]
   기온 14.8도 → 검색어: 기온, 평균기온
   출생아수 6.7% 증가 → 검색어: 출생아수, 출생
   혼인 건수 → 검색어: 혼인건수, 혼인
   대졸이상 청년 쉬었음 비율 → 검색어: 쉬었음, 비경제활동인구
                              (대졸이상, 청년은 분류 축 — 넣지 마세요)
   여성 65세 1인가구 → 검색어: 1인가구, 가구

   [잘못된 예시]
   ❌ "대졸이상 청년 쉬었음 비율, 청년 교육 수준별 실업률"  (분류 축 + 합성어 + 너무 김)
   ❌ "2024년 연평균 기온 통계"  (연도, 통계 같은 군더더기)
   ❌ "월별 출생아수 변화 추이"   (수식어 "변화 추이")

[중요 출력 규칙]
- 마크다운 강조(*, _, `, #, ~) 절대 사용 금지. 일반 텍스트만.
- "카테고리:"와 "검색어:" 두 줄, 각 줄에 실제 한국어 단어만.
- indicator가 비어있어도 일반적 카테고리(예: 인구, 경제)로 답하세요.

형식:
카테고리: 키워드1, 키워드2, 키워드3
검색어: 키워드1, 키워드2"""


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
        # [v6.6 변경] 임베딩 텍스트 포맷을 박재윤 factcheck_test.py v6 기준으로 강화
        # KOSIS 카탈로그가 인덱싱될 때 사용된 포맷과 매칭되도록:
        #   "{categories} > {indicator} | 항목: {indicator} | 분류: {kw} | 단위: {unit}"
        # 카테고리는 catalog_search 내부 LLM이 방금 추출했으므로 여기서 합쳐줘야 함.
        # query_builder는 indicator/population/unit만 넘김 → 여기서 cats + 분류(search_kw) 추가.
        #
        # [v6.5 기존 — 비교용으로 주석 보존]
        # embedding_text = (query.extra_params or {}).get("embedding_text") or (
        #     " ".join(filter(None, [query.indicator, query.population, search_kw]))
        # )
        base_emb = (query.extra_params or {}).get("embedding_text") or (
            " ".join(filter(None, [query.indicator, query.population, search_kw]))
        )
        cats_str = " ".join(category_kws) if category_kws else ""
        # 카테고리 + 분류 keyword를 base_emb에 결합 (KOSIS 카탈로그 포맷)
        embedding_text = (
            f"{cats_str} > {base_emb} | 분류: {search_kw}"
        ).strip()

        # [v6.6 추가] stat_name 토큰 추출 — pgvector WHERE에 stat_name ILIKE 필터로 활용
        # 박재윤 factcheck_test.py의 field_terms와 동일한 로직.
        # 출생아 수, 혼인 건수 같은 단어가 stat_name에 직접 매칭되는 표를 끌어와야 함.
        stopwords = {"수", "명", "원", "건", "개", "만", "천", "억", "이상", "이하", "약", "총", "전체"}
        field_terms: list[str] = []
        if query.indicator:
            for tok in re.split(r"[\s·,/]+", query.indicator):
                if len(tok) >= 2 and tok not in stopwords and not re.match(r"^[\d.]+$", tok):
                    field_terms.append(tok)

        embedding = await self._get_embedding(embedding_text)

        if embedding:
            # 2) category 필터 + (NEW) stat_name 필터 + embedding
            if category_kws or field_terms:
                cat_recs = await self._search_pgvector(
                    embedding,
                    category_keywords=category_kws,
                    field_name_terms=field_terms,   # [v6.6 추가]
                    top_k=top_k,
                )
                _add(cat_recs)

            # 3) 전체 embedding (폴백) — 변경 없음
            all_recs = await self._search_pgvector(embedding, category_keywords=None, top_k=5)
            _add(all_recs)

        final_results = results[:top_k]
        logger.info(f"CatalogSearch 완료: {len(results)}개 후보")
        # [v6.5 진단] 후보 stat_id 전체를 로그에 — 어떤 표가 검색되고 어떤 게 누락되는지 추적
        if final_results:
            stat_id_preview = ", ".join(
                f"[{r.stat_id}]{r.stat_name[:20]}" for r in final_results
            )
            logger.info(f"CatalogSearch 후보들: {stat_id_preview}")
        return final_results

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
                    _strip_markdown(c) for c in cats.split(",") if c.strip()
                ]
                # 빈 문자열 제거 (markdown만 있던 항목 처리)
                category_keywords = [c for c in category_keywords if c]
            elif "검색어" in line and ":" in line:
                kw = line.split(":", 1)[1].strip().strip("\"'")
                kw = _strip_markdown(kw)
                kw = re.sub(r"\b\d{4}\b", "", kw).strip()
                kw = re.sub(r"\s+", " ", kw)
                if kw:
                    search_keyword = kw

        # [v6.3] 모든 정제 후에도 키워드가 비어있거나 너무 짧으면 fallback
        # 또한 한글/영문 글자가 하나도 없으면(특수문자만) fallback
        clean_check = re.sub(r"[^\w가-힣]", "", search_keyword)
        if not search_keyword.strip() or len(clean_check) < 2:
            search_keyword = query.indicator or query.keyword or ""
            logger.debug(f"검색어 추출 결과 부적합 → fallback: {search_keyword!r}")

        # category도 같은 검사
        category_keywords = [
            c for c in category_keywords
            if len(re.sub(r"[^\w가-힣]", "", c)) >= 2
        ]

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
        field_name_terms: list[str] | None = None,   # [v6.6 추가]
        top_k: int = 5,
    ) -> list[StatRecord]:
        """
        kosis_stat_catalog pgvector 유사도 검색.

        [v6.6 변경] field_name_terms 옵션 추가 — stat_name ILIKE 필터.
        category_path와 stat_name 양쪽에서 매칭되는 것을 모두 끌어옴 (OR).
        박재윤 factcheck_test.py v6 search_pgvector 로직과 동일.

        category_keywords + field_name_terms 둘 다 있으면: (cat OR stat_name) AND embedding 정렬
        하나만 있으면: 그것 OR + embedding 정렬
        둘 다 없으면: 전체 embedding 정렬
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
            # [v6.6] WHERE 절을 동적으로 구성 — category OR stat_name 둘 다 검사
            # [v6.5 기존 — 비교용 주석]
            # if category_keywords:
            #     where_parts = [f"category_path ILIKE ${i+2}" for i in range(len(category_keywords))]
            #     where_sql   = " OR ".join(where_parts)
            #     params      = [vector_str] + [f"%{kw}%" for kw in category_keywords] + [top_k]
            where_clauses: list[str] = []
            ilike_params: list[str] = []
            param_idx = 2  # $1은 vector

            if category_keywords:
                cat_parts = [f"category_path ILIKE ${param_idx + i}" for i in range(len(category_keywords))]
                where_clauses.append("(" + " OR ".join(cat_parts) + ")")
                ilike_params.extend([f"%{kw}%" for kw in category_keywords])
                param_idx += len(category_keywords)

            if field_name_terms:
                name_parts = [f"stat_name ILIKE ${param_idx + i}" for i in range(len(field_name_terms))]
                where_clauses.append("(" + " OR ".join(name_parts) + ")")
                ilike_params.extend([f"%{t}%" for t in field_name_terms])
                param_idx += len(field_name_terms)

            if where_clauses:
                where_sql = " OR ".join(where_clauses)
                params = [vector_str] + ilike_params + [top_k]
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

            label_parts = []
            if category_keywords:
                label_parts.append(f"cat={category_keywords}")
            if field_name_terms:
                label_parts.append(f"name={field_name_terms}")
            label = "(" + ", ".join(label_parts) + ")" if label_parts else "(전체)"
            logger.debug(f"pgvector 검색 {label}: {len(records)}개")
            return records

        except Exception as e:
            logger.warning(f"pgvector 검색 실패: {e}")
            try:
                await conn.close()
            except Exception:
                pass
            return []