# [2026-05-14 | 이수민] memory/v1: KOSIS API + pgvector 결과 metadata 머지
#   - 같은 stat_id가 두 경로에서 모두 나올 때 metadata 머지
#   - KOSIS API 결과(먼저 추가)에 pgvector 결과의 category_path를 보강
#   - 도메인 가드 인프라용 (verifier에서 evidence.category_path 사용)
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


# ── [v6.21] 수록주기 인지 재정렬 ──────────────────────────────────────
# catalog 인덱스(kosis_stat_catalog)에 수록주기 컬럼이 없어, 표 이름
# 텍스트에서 추론한다. claim이 월/분기 시점인데 검색 결과 상위가 전부
# 연 단위 표면 fetch가 모두 실패하므로(표 시점 단위 불일치 거부),
# 월/분기 claim일 때 월간 신호가 있는 표를 앞으로 끌어올린다.
# 도메인 무관: 표 이름의 보편적 주기 어휘만 사용 (인구·기온·고용 공통).
_MONTHLY_NAME_HINTS = ("월별", "월간", "월.", "월·", "/월", "매월", "월말")
_YEARLY_NAME_HINTS = ("연간", "연도별", "연도말", "장래", "추계", "년간")


def _claim_period_unit(time_period: str | None) -> str:
    """claim time_period의 단위 판별: month / quarter / year / unknown."""
    if not time_period:
        return "unknown"
    s = str(time_period).strip()
    # YYYY-MM, YYYY.MM, YYYYMM → month
    if re.match(r"^\d{4}[-./]?\d{2}$", s):
        return "month"
    # YYYY-Q1 등 분기
    if re.search(r"[Qq][1-4]", s) or re.match(r"^\d{4}[-./]?[1-4]$", s):
        return "quarter"
    if re.match(r"^\d{4}$", s):
        return "year"
    return "unknown"


def _name_periodicity(stat_name: str) -> str:
    """표 이름 텍스트에서 수록주기 추론: monthly / yearly / mixed / unknown.

    "월.분기.연간 인구동향" → mixed (월 신호 있음 → 월 claim에 사용 가능)
    "장래 합계출산율"        → yearly
    "시도/혼인종류별 혼인"    → unknown (주기 표기 없음)
    """
    if not stat_name:
        return "unknown"
    s = str(stat_name)
    has_month = any(h in s for h in _MONTHLY_NAME_HINTS)
    has_year = any(h in s for h in _YEARLY_NAME_HINTS)
    if has_month and has_year:
        return "mixed"   # "월.분기.연간" — 월 데이터 포함
    if has_month:
        return "monthly"
    if has_year:
        return "yearly"
    return "unknown"


def _rerank_by_periodicity(
    records: list[StatRecord], claim_unit: str
) -> list[StatRecord]:
    """claim 시점 단위에 맞춰 후보를 재정렬.

    claim이 월/분기인데 상위가 전부 연 단위 표면 검증이 불가능하므로,
    월간 신호가 있는 표(monthly/mixed)를 앞으로, 명백한 연 단위 표
    (yearly)를 뒤로 보낸다. embedding 유사도 순서는 같은 등급 안에서 유지.

    claim이 연 단위거나 unknown이면 원순서 그대로 (가드 불필요).
    """
    if claim_unit not in ("month", "quarter"):
        return records

    def _priority(r: StatRecord) -> int:
        p = _name_periodicity(getattr(r, "stat_name", "") or "")
        if p in ("monthly", "mixed"):
            return 0   # 월 데이터 있음 — 최우선
        if p == "unknown":
            return 1   # 판별 불가 — 중립
        return 2       # yearly — 월 claim엔 부적합, 후순위

    # stable sort: 같은 우선순위는 기존(유사도) 순서 유지
    return sorted(records, key=_priority)

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
        id_to_rec: dict[str, StatRecord] = {}  # [이수민 2026-05-14] metadata 머지용

        def _add(recs: list[StatRecord]) -> None:
            for r in recs:
                if r.stat_id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.stat_id)
                    id_to_rec[r.stat_id] = r
                else:
                    # [이수민 2026-05-14] 같은 stat_id 중복: metadata 머지
                    # KOSIS API 결과(먼저)는 category_path 없고 pgvector 결과(나중)는 있음
                    # → category_path를 KOSIS API 결과에 채워줌
                    existing = id_to_rec.get(r.stat_id)
                    if existing and not existing.metadata.get("category_path"):
                        cp = r.metadata.get("category_path")
                        if cp:
                            existing.metadata["category_path"] = cp

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
            # [P29' 2026-05-22] top_k 5 → 15. category 필터(2)가 KOSIS 자체 어휘랑
            # 안 맞으면 (예: LLM이 자유어 '의료기기' 추출했는데 KOSIS category_path는
            # '보건 > 의료자원 > 의료장비') (2)가 0건이라 정답이 (3)에 의존하는데, 그
            # (3)이 top_k=5라 cosine 6~15위인 정답 표가 잘림. row-level keyword가
            # 표 이름에 없는 케이스(예: "체외 충격파 쇄석술 장비")는 점수가 어차피
            # 0.5~0.7 수준이라 더 많이 받아야 정답 진입.
            all_recs = await self._search_pgvector(embedding, category_keywords=None, top_k=15)
            _add(all_recs)

        # [v6.21] 수록주기 인지 재정렬 — claim이 월/분기 시점이면
        # 월간 데이터가 있는 표를 상위로. 연 단위 표만 상위에 오면
        # fetch가 모두 '표 시점 단위 불일치'로 거부돼 검증 불가가 된다.
        claim_unit = _claim_period_unit(getattr(query, "time_period", None))
        if claim_unit in ("month", "quarter"):
            before = [r.stat_id for r in results[:3]]
            results = _rerank_by_periodicity(results, claim_unit)
            after = [r.stat_id for r in results[:3]]
            if before != after:
                logger.info(
                    f"CatalogSearch 수록주기 재정렬 (claim={claim_unit}): "
                    f"top3 {before} → {after}"
                )

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
        # [2026-05-26 회귀] 옵션 B(parent_path + LLM union)가 LLM 자유 추출의 노이즈
        # 때문에 catalog 결과 망가뜨림 — 옛 동작(parent_path 우선, LLM은 fallback만)으로
        # 되돌림. parent_path 있으면 그대로 분해해 사용하고 LLM 호출 안 함.
        parent_path = (query.extra_params or {}).get("parent_path")
        if parent_path:
            parts = [p.strip() for p in re.split(r"\s*>\s*", parent_path) if p.strip()]
            if parts:
                search_kw = _minimal_clean(parts[-1])
                _raw_cats = parts[:-1] if len(parts) > 1 else parts
                category_kws = [
                    _minimal_clean(c) for c in _raw_cats if _minimal_clean(c)
                ]
                if search_kw:
                    logger.debug(f"parent_path 활용: category={category_kws}, kw={search_kw}")
                    return (category_kws, search_kw)

        # parent_path 없을 때만 LLM 호출 (fallback)
        path_category_kws: list[str] = []
        path_search_kw = ""
        llm_category_kws: list[str] = []
        llm_search_kw = ""
        if self.hcx_key:

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
                for line in content.split("\n"):
                    line = line.strip()
                    if "카테고리" in line and ":" in line:
                        cats = line.split(":", 1)[1].strip()
                        llm_category_kws = [
                            _minimal_clean(c) for c in cats.split(",")
                            if _minimal_clean(c)
                        ]
                    elif "검색어" in line and ":" in line:
                        kw = line.split(":", 1)[1].strip().strip("\"'")
                        kw = _minimal_clean(kw)
                        if kw:
                            llm_search_kw = kw
            except Exception as e:
                logger.debug(f"카테고리 LLM 추출 실패 (path category로 폴백): {e}")

        # 3) union 결합 — path 우선, LLM은 보강용
        # [2026-05-26 보정] LLM 자유 추출이 가끔 쉼표 풀어쓴 긴 문자열(예: '체외충격파
        # 쇄석술장비수, 강원도의료기기현황')을 search_kw로 박거나, KOSIS 분야와 어긋난
        # 카테고리(예: '의료기기', '지역통계')를 만들어서 catalog 검색이 완전 무관한
        # 표(장애인 거주시설, 노숙인 시설 등)를 잡아오는 사고 발생. 그래서:
        #   - search_kw: path_search_kw 우선 (단순 단어), LLM은 fallback 또는 단어 1개
        #   - category: path + LLM 한 단어짜리만 union (긴 풀어쓰기 거부)

        def _is_clean_token(s: str) -> bool:
            """검색어/카테고리로 안전한 단순 토큰인지. 쉼표/공백 6자 이상 + 너무 길면 거부."""
            if not s:
                return False
            if "," in s or "、" in s:
                return False  # 쉼표 풀어쓰기는 의도 모호
            if len(s) > 20:
                return False  # 너무 긴 자유 텍스트는 검색 노이즈
            return True

        # category: path가 우선, LLM은 *깨끗한 토큰*만 추가
        _clean_llm_cats = [c for c in llm_category_kws if _is_clean_token(c)]
        category_keywords: list[str] = list(dict.fromkeys(
            [*path_category_kws, *_clean_llm_cats]
        ))[:5]

        # search_kw: path 우선, LLM은 깨끗할 때만 fallback
        if path_search_kw and _is_clean_token(path_search_kw):
            search_keyword = path_search_kw
        elif llm_search_kw and _is_clean_token(llm_search_kw):
            search_keyword = llm_search_kw
        else:
            search_keyword = _minimal_clean(query.indicator or query.keyword or "")
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
    """LLM 응답에서 명백한 노이즈만 제거 (연도 + 마크다운 + 공백).

    의미 정제는 LLM 책임이지만, planner LLM이 검색어/카테고리에
    마크다운 강조(**, *, `, #)를 섞어 출력하는 경우가 잦아
    (예: '** 연평균 기온') 검색을 오염시키므로 여기서 제거한다.
    """
    if not kw:
        return ""
    s = str(kw)
    # 마크다운 강조/머리표 제거: **bold**, *italic*, `code`, # heading, - bullet
    s = s.replace("*", "").replace("`", "").replace("#", "")
    s = re.sub(r"^\s*[-•·]\s*", "", s)   # 줄머리 불릿
    s = re.sub(r"\b\d{4}\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(",·\"' -")