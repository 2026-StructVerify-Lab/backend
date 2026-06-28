"""
structverify.agent.tools.explore_catalog — 카탈로그 어휘 탐색 Tool.

용도: LLM이 어떤 카테고리 어휘를 써야 할지 모를 때 우선 호출.
     KOSIS 카탈로그가 실제로 쓰는 분류명/대표 표를 보여줘서,
     룰베이스 도메인 매핑 없이도 LLM이 정확한 catalog_search query/category를
     만들 수 있도록 self-learning 시킨다.

[패치 R1] 검색을 **임베딩 기반**으로 재구현. 기존 ILIKE 키워드 매칭은
     '연평균 기온' 같은 합성어를 못 잡아 적합 카테고리('기상관측통계') 누락.
     이제 query embedding → pgvector 거리 정렬로 top N 표 → 그 표들의
     category_path 분포 집계. 의미 매칭이므로 키워드가 표 이름에 직접
     안 들어가도 가까운 카테고리를 정확히 찾는다.

예시:
  query="연평균 기온"
  → query embedding과 가까운 top 100 표를 거리 정렬
    → 그 표들이 속한 카테고리 집계:
       1. 기상관측통계 (45개)  [DT_14102_B001 [종관기상] ..., DT_14104_N_002 ...]
       2. 환경 (12개)           [DT_2OEEG008 연평균 기온 변화, ...]
       3. ...
  → LLM이 정확한 cat_label로 catalog_search 호출
"""
from __future__ import annotations

import os
from typing import Any
from collections import defaultdict
from structverify.utils.logger import get_logger
from structverify.utils.embedding_client import EmbeddingClient

from ..schemas import ActionType
from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


@register_tool(ActionType.EXPLORE_CATALOG)
class ExploreCatalogTool(ToolBase):
    """카탈로그 카테고리 + 대표 표 탐색 — LLM의 self-learning용 (임베딩 기반)."""

    name = ActionType.EXPLORE_CATALOG
    description = (
        "카탈로그의 카테고리 분포 + 대표 표 미리보기 (임베딩 의미 검색). "
        "어떤 분류 어휘를 써야 할지 모를 때 catalog_search 전에 먼저 호출. "
        "결과의 category_label을 catalog_search의 category 인자에 그대로 넣어라."
    )
    input_schema = {
        "query": "(선택) 관심 주제 키워드. 비우면 전체 대분류 분포.",
        "top_categories": "(선택) 보여줄 카테고리 수. 기본 5",
        "examples_per_category": "(선택) 카테고리당 대표 표 수. 기본 2",
        "scan_pool": "(선택) 임베딩 검색 풀 크기. 기본 100",
    }

    # 임베딩 helper 캐시 (CatalogSearchTool 인스턴스 재사용)
    _embedding_helper = None

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query = (input_data.get("query") or "").strip()
        # [#67-D] context.config.embedding 을 _embed_query가 쓰도록 self에 보관.
        self._embedding_config = (context.config or {}).get("embedding") or {}

        def _safe_int(k, default, lo, hi):
            try:
                v = int(input_data.get(k) or default)
            except (TypeError, ValueError):
                v = default
            return max(lo, min(v, hi))

        top_categories = _safe_int("top_categories", 5, 1, 15)
        examples_per = _safe_int("examples_per_category", 2, 1, 5)
        scan_pool = _safe_int("scan_pool", 100, 20, 300)

        try:
            import asyncpg
        except ImportError:
            return ToolResult(
                output={},
                summary="실패: asyncpg 미설치 — pgvector 탐색 불가",
                success=False,
                error="asyncpg not installed",
            )

        pg_dsn = os.environ.get(
            "PGVECTOR_DSN",
            "postgresql://structverify:svpass123@localhost:5432/structverify",
        )

        try:
            conn = await asyncpg.connect(pg_dsn)
        except Exception as e:
            logger.warning(f"[explore_catalog] DB 연결 실패: {e}")
            return ToolResult(
                output={},
                summary=f"실패: 카탈로그 DB 연결 — {type(e).__name__}: {e}",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        try:
            if query:
                categories_info = await self._fetch_by_embedding(
                    conn, query, top_categories, examples_per, scan_pool
                )
                used_method = "embedding"
                # 임베딩 실패 시 ILIKE fallback
                if not categories_info:
                    logger.info(
                        f"[explore_catalog] embedding 결과 없음 → ILIKE fallback"
                    )
                    categories_info = await self._fetch_by_ilike(
                        conn, query, top_categories, examples_per
                    )
                    used_method = "ilike_fallback"
            else:
                # query 없으면 전체 대분류 분포
                categories_info = await self._fetch_all_categories(
                    conn, top_categories, examples_per
                )
                used_method = "overview"
        except Exception as e:
            logger.exception(f"[explore_catalog] 쿼리 실패: query={query!r}")
            try:
                await conn.close()
            except Exception:
                pass
            return ToolResult(
                output={},
                summary=f"실패: 카탈로그 쿼리 — {type(e).__name__}: {e}",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            try:
                await conn.close()
            except Exception:
                pass

        # LLM 친화 포맷 — sim 값은 (정규화되지 않은) 임베딩이라 절대값이 큰 음수로
        # 표시될 수 있어 LLM에 혼란. 순위만 보여주고 sim은 숨김.
        lines: list[str] = []
        for idx, info in enumerate(categories_info, start=1):
            cat_label = info["category_label"]
            count = info["table_count"]
            lines.append(f"{idx}. {cat_label} ({count}개 표)")
            for ex in info["examples"]:
                lines.append(f"   - [{ex['stat_id']}] {ex['stat_name']}")
        summary_block = "\n".join(lines) if lines else "(카탈로그에 매칭 카테고리 없음)"

        # [R1.5] 다음 단계 명령을 *복사 가능한 JSON 형식*으로 명시.
        # LLM이 자기 어휘로 catalog_search category를 만드는 걸 방지.
        # (시스템은 R1.5 패치로 자동 union 하지만, 프롬프트도 강화.)
        top_labels = [info["category_label"] for info in categories_info[:2]]
        if top_labels:
            cat_json = '["' + '", "'.join(top_labels) + '"]'
            next_step = (
                f"→ 다음 단계 (필수): catalog_search 호출 시 input.category 인자에 "
                f"위 1·2위 카테고리를 *그대로 복사*해서 사용하라:\n"
                f'   input.category = {cat_json}\n'
                f"   query는 claim의 핵심 지표명(예: indicator)을 그대로 쓸 것. "
                f"임의의 자유어('기후 변화', '날씨 정보' 같은)는 사용 금지."
            )
        else:
            next_step = (
                "→ 다음 단계: explore_catalog를 더 specific한 query로 재시도하거나, "
                "catalog_search에 query만 넣고 호출 (category 없이)."
            )
        summary_head = (
            f"explore_catalog(query={query!r}, method={used_method}): "
            f"{len(categories_info)}개 카테고리\n{summary_block}\n\n{next_step}"
        )

        # observation 저장
        try:
            obs_name = f"iter{context.iter_num:03d}_explore_catalog"
            context.workspace.write_observation(
                context.claim_id,
                obs_name,
                {
                    "query": query,
                    "method": used_method,
                    "categories": categories_info,
                },
            )
        except Exception as e:
            logger.debug(f"[explore_catalog] observation 저장 실패: {e}")

        return ToolResult(
            output={
                "query": query,
                "method": used_method,
                "categories": categories_info,
                "category_count": len(categories_info),
            },
            summary=summary_head,
            success=True,
        )

    # ── 임베딩 기반 탐색 (주 경로) ──────────────────────────────────

    async def _fetch_by_embedding(
        self,
        conn,
        query: str,
        top_categories: int,
        examples_per: int,
        scan_pool: int,
    ) -> list[dict[str, Any]]:
        """query 임베딩 → pgvector 거리 정렬 → top N 표의 카테고리 분포 집계."""
        embedding = await self._embed_query(query)
        if embedding is None:
            logger.info(
                f"[explore_catalog] 임베딩 생성 실패(API/key 문제) — ILIKE fallback 예정"
            )
            return []

        vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
        rows = await conn.fetch(
            """
            SELECT stat_id, stat_name, category_path,
                   1 - (embedding <-> $1::vector) AS similarity
            FROM kosis_stat_catalog
            WHERE embedding IS NOT NULL
              AND category_path IS NOT NULL
            ORDER BY embedding <-> $1::vector
            LIMIT $2
            """,
            vector_str, scan_pool,
        )

        # 카테고리별 그룹화 (category_label = path의 첫 의미 segment)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            path = r["category_path"]
            cat_label = self._cat_label_from_path(path)
            if not cat_label:
                continue
            buckets[cat_label].append({
                "stat_id": r["stat_id"],
                "stat_name": r["stat_name"],
                "similarity": float(r["similarity"]),
            })

        # 카테고리 정렬: 그 안의 평균 similarity 내림차순
        scored: list[tuple[str, int, float, list[dict[str, Any]]]] = []
        for cat, tables in buckets.items():
            avg_sim = sum(t["similarity"] for t in tables) / len(tables)
            # 표는 similarity 내림차순으로 정렬
            tables.sort(key=lambda t: t["similarity"], reverse=True)
            scored.append((cat, len(tables), avg_sim, tables))
        scored.sort(key=lambda x: x[2], reverse=True)

        result: list[dict[str, Any]] = []
        for cat, cnt, avg_sim, tables in scored[:top_categories]:
            examples = [
                {"stat_id": t["stat_id"], "stat_name": t["stat_name"]}
                for t in tables[:examples_per]
            ]
            result.append({
                "category_label": cat,
                "table_count": cnt,  # scan_pool 내 카운트
                "avg_similarity": round(avg_sim, 3),
                "examples": examples,
            })
        return result

    @staticmethod
    def _cat_label_from_path(path: str) -> str:
        """category_path → 의미 있는 첫 segment.

        'MT_ZTITLE > 기상관측통계 > 종관기상' → '기상관측통계'
        'MT_ZTITLE > 인구동향조사' → '인구동향조사'
        '기상관측통계' (단독) → '기상관측통계'
        """
        if not path:
            return ""
        parts = [p.strip() for p in path.split(" > ") if p.strip()]
        if not parts:
            return ""
        # 첫 segment가 메타 prefix면 다음 사용
        _META_PREFIXES = ("MT_ZTITLE", "MT_OTITLE")
        if parts[0] in _META_PREFIXES and len(parts) > 1:
            return parts[1]
        return parts[0]

    # ── 임베딩 helper ──────────────────────────────────────────────

    async def _embed_query(self, text: str) -> list[float] | None:
        """텍스트 → 임베딩. 공용 EmbeddingClient 사용 (#67-D).

        키 우선순위: 원래 NCP_API_KEY 우선이라 api_key_env 기본을 "NCP_API_KEY"로 둠.
        config.embedding.api_key_env 가 있으면 그 값을 우선.
        주의: 원래 NCP_API_KEY or CLOVASTUDIO_API_KEY 폴백이었으나 EmbeddingClient는
        단일 api_key_env → NCP만 1순위(NCP 없을 때 CLOVA 폴백은 제거됨).
        """
        emb_cfg = dict(getattr(self, "_embedding_config", None) or {})
        emb_cfg.setdefault("api_key_env", "NCP_API_KEY")
        try:
            return await EmbeddingClient(emb_cfg).embed(text)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[explore_catalog] _embed_query 실패: {e}")
            return None

    # ── ILIKE fallback (임베딩 실패 / 0건 결과 시) ──────────────────

    async def _fetch_by_ilike(
        self,
        conn,
        query: str,
        top_categories: int,
        examples_per: int,
    ) -> list[dict[str, Any]]:
        """기존 ILIKE 키워드 매칭 — 임베딩 fallback용."""
        tokens = [t for t in query.split() if len(t) >= 2] or [query]
        where_parts: list[str] = []
        params: list[Any] = []
        for t in tokens:
            params.append(f"%{t}%")
            idx_a = len(params)
            where_parts.append(
                f"(stat_name ILIKE ${idx_a} OR category_path ILIKE ${idx_a})"
            )
        where_sql = " OR ".join(where_parts)
        params.append(top_categories)
        limit_idx = len(params)
        sql = f"""
            SELECT
                COALESCE(NULLIF(split_part(category_path, ' > ', 2), ''),
                         category_path) AS cat_label,
                COUNT(*) AS cnt
            FROM kosis_stat_catalog
            WHERE ({where_sql})
              AND category_path IS NOT NULL
            GROUP BY cat_label
            ORDER BY cnt DESC
            LIMIT ${limit_idx}
        """
        rows = await conn.fetch(sql, *params)

        result: list[dict[str, Any]] = []
        for row in rows:
            cat_label = row["cat_label"]
            cnt = int(row["cnt"])
            # 대표 표 — 같은 ILIKE 매칭으로 fetch
            ex_where: list[str] = []
            ex_params: list[Any] = [cat_label]
            for t in tokens:
                ex_params.append(f"%{t}%")
                ex_where.append(
                    f"(stat_name ILIKE ${len(ex_params)} OR category_path ILIKE ${len(ex_params)})"
                )
            ex_params.append(examples_per)
            ex_sql = f"""
                SELECT stat_id, stat_name
                FROM kosis_stat_catalog
                WHERE COALESCE(NULLIF(split_part(category_path, ' > ', 2), ''),
                               category_path) = $1
                  AND ({" OR ".join(ex_where)})
                ORDER BY stat_name
                LIMIT ${len(ex_params)}
            """
            ex_rows = await conn.fetch(ex_sql, *ex_params)
            examples = [
                {"stat_id": r["stat_id"], "stat_name": r["stat_name"]}
                for r in ex_rows
            ]
            result.append({
                "category_label": cat_label,
                "table_count": cnt,
                "examples": examples,
            })
        return result

    # ── query 없을 때 전체 대분류 분포 ────────────────────────────

    async def _fetch_all_categories(
        self,
        conn,
        top_categories: int,
        examples_per: int,
    ) -> list[dict[str, Any]]:
        rows = await conn.fetch(
            """
            SELECT
                COALESCE(NULLIF(split_part(category_path, ' > ', 2), ''),
                         category_path) AS cat_label,
                COUNT(*) AS cnt
            FROM kosis_stat_catalog
            WHERE category_path IS NOT NULL
            GROUP BY cat_label
            ORDER BY cnt DESC
            LIMIT $1
            """,
            top_categories,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            cat_label = row["cat_label"]
            cnt = int(row["cnt"])
            ex_rows = await conn.fetch(
                """
                SELECT stat_id, stat_name
                FROM kosis_stat_catalog
                WHERE COALESCE(NULLIF(split_part(category_path, ' > ', 2), ''),
                               category_path) = $1
                ORDER BY stat_name
                LIMIT $2
                """,
                cat_label, examples_per,
            )
            result.append({
                "category_label": cat_label,
                "table_count": cnt,
                "examples": [
                    {"stat_id": r["stat_id"], "stat_name": r["stat_name"]}
                    for r in ex_rows
                ],
            })
        return result
