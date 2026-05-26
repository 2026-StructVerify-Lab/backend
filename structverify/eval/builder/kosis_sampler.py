"""KOSIS catalog DB sampling for eval gold construction."""
from __future__ import annotations

import os
from typing import Any

from structverify.eval.builder.domain_mapping import get_category_keywords
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_pg_dsn(config: dict | None = None) -> str:
    """
    Catalog DB DSN — pipeline catalog_search와 동일 DB.

    우선순위:
      1) PGVECTOR_DSN (또는 eval_builder kosis.pgvector_dsn_env)
      2) POSTGRES_DSN
      3) POSTGRES_HOST/PORT/DB/USER/PASSWORD 조합 (db_manager·kosis_crawler와 동일)
    """
    from dotenv import load_dotenv

    load_dotenv()

    cfg = config or {}
    kosis_cfg = cfg.get("kosis", {})
    env_name = kosis_cfg.get("pgvector_dsn_env", "PGVECTOR_DSN")

    dsn = os.environ.get(env_name) or os.environ.get("POSTGRES_DSN")
    if dsn:
        return dsn

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "structverify")
    user = os.getenv("POSTGRES_USER", "structverify")
    password = os.getenv("POSTGRES_PASSWORD")
    if password is not None and password != "":
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    logger.warning(
        "PGVECTOR_DSN/POSTGRES_DSN/POSTGRES_PASSWORD 없음 → "
        "로컬 기본 DSN 사용 (docker-compose 기본값). .env 확인 권장."
    )
    return "postgresql://structverify:svpass123@localhost:5432/structverify"


class KosisRowSampler:
    """Stratified sampling from kosis_stat_catalog."""

    def __init__(self, config: dict | None = None, seed: int = 42):
        self.config = config or {}
        self.seed = seed
        self.pg_dsn = resolve_pg_dsn(self.config)
        self.max_attempts = int(
            self.config.get("kosis", {}).get("max_sample_attempts_per_claim", 8)
        )

    async def _connect(self):
        import asyncpg

        return await asyncpg.connect(self.pg_dsn)

    async def count_catalog_rows(self) -> int:
        conn = await self._connect()
        try:
            row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM kosis_stat_catalog")
            return int(row["cnt"]) if row else 0
        finally:
            await conn.close()

    async def scan_domain_density(self, domains: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        conn = await self._connect()
        try:
            for domain in domains:
                keywords = get_category_keywords(domain)
                if keywords:
                    params = [f"%{kw}%" for kw in keywords]
                    cat_parts = [
                        f"category_path ILIKE ${i + 1}" for i in range(len(keywords))
                    ]
                    where_sql = f"({' OR '.join(cat_parts)})"
                else:
                    where_sql = "TRUE"
                    params = []
                sql = f"SELECT COUNT(*) AS cnt FROM kosis_stat_catalog WHERE {where_sql}"
                row = await conn.fetchrow(sql, *params)
                counts[domain] = int(row["cnt"]) if row else 0
        finally:
            await conn.close()
        logger.info(f"Domain density scan: {counts}")
        return counts

    @staticmethod
    def _distribute_integer_quotas(
        domains: list[str],
        total_articles: int,
        weights: dict[str, float],
    ) -> dict[str, int]:
        """Largest-remainder allocation so article counts sum to total_articles."""
        positive = {d: max(float(weights.get(d, 0)), 0.0) for d in domains}
        positive = {d: w for d, w in positive.items() if w > 0}
        if not positive:
            return {d: 0 for d in domains}
        total_weight = sum(positive.values()) or 1.0
        raw = {d: total_articles * positive[d] / total_weight for d in positive}
        quotas = {d: int(raw[d]) for d in positive}
        remainder = total_articles - sum(quotas.values())
        order = sorted(
            positive.keys(),
            key=lambda d: (raw[d] - quotas[d], positive[d]),
            reverse=True,
        )
        idx = 0
        while remainder > 0 and order:
            quotas[order[idx % len(order)]] += 1
            remainder -= 1
            idx += 1
        return quotas

    @staticmethod
    def allocate_domain_quotas(
        domains: list[str],
        total_articles: int,
        density: dict[str, int],
        domain_shares: dict[str, float] | None = None,
        domain_articles: dict[str, int] | None = None,
    ) -> dict[str, int]:
        if domain_articles:
            filtered = {d: int(domain_articles[d]) for d in domains if d in domain_articles}
            if filtered:
                return filtered

        if domain_shares:
            weights = {d: float(domain_shares[d]) for d in domains if d in domain_shares}
            if weights:
                return KosisRowSampler._distribute_integer_quotas(
                    domains, total_articles, weights
                )

        # catalog 0건 도메인도 최소 1슬롯 — finance/weather 등 완전 누락 방지
        positive = {d: max(density.get(d, 0), 1) for d in domains}
        return KosisRowSampler._distribute_integer_quotas(
            domains, total_articles, positive
        )

    async def sample_catalog_row(
        self,
        domain: str,
        exclude_stat_ids: set[str] | None = None,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        exclude_stat_ids = exclude_stat_ids or set()
        keywords = get_category_keywords(domain)
        conn = await self._connect()
        try:
            params: list[Any] = []
            if keywords:
                cat_parts = []
                for kw in keywords:
                    params.append(f"%{kw}%")
                    cat_parts.append(f"category_path ILIKE ${len(params)}")
                where_sql = f"({' OR '.join(cat_parts)})"
            else:
                where_sql = "TRUE"

            fetch_limit = limit
            params.append(str(self.seed))
            seed_idx = len(params)
            params.append(fetch_limit)
            limit_idx = len(params)

            if exclude_stat_ids:
                params.append(list(exclude_stat_ids))
                ex_idx = len(params)
                sql = f"""
                    SELECT stat_id, stat_name, org_id, org_name, category_path, keywords
                    FROM kosis_stat_catalog
                    WHERE {where_sql}
                      AND NOT (stat_id = ANY(${ex_idx}::text[]))
                    ORDER BY md5(stat_id || ${seed_idx}::text)
                    LIMIT ${limit_idx}
                """
            else:
                sql = f"""
                    SELECT stat_id, stat_name, org_id, org_name, category_path, keywords
                    FROM kosis_stat_catalog
                    WHERE {where_sql}
                    ORDER BY md5(stat_id || ${seed_idx}::text)
                    LIMIT ${limit_idx}
                """
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def sample_candidates_for_domain(
        self,
        domain: str,
        exclude_facts: set[tuple[str, str]],
        exclude_stat_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to max_attempts distinct catalog rows (single DB round-trip)."""
        used_stats = {stat_id for stat_id, _ in exclude_facts}
        if exclude_stat_ids:
            used_stats |= exclude_stat_ids

        candidates = await self.sample_catalog_row(
            domain,
            exclude_stat_ids=used_stats,
            limit=self.max_attempts,
        )
        if not candidates:
            keywords = get_category_keywords(domain)
            logger.warning(
                f"No catalog candidates for domain={domain} keywords={keywords} "
                f"(excluded {len(used_stats)} stat_ids)"
            )
        return candidates

    async def sample_for_domain(
        self,
        domain: str,
        exclude_facts: set[tuple[str, str]],
        exclude_stat_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        candidates = await self.sample_candidates_for_domain(
            domain, exclude_facts, exclude_stat_ids
        )
        return candidates[0] if candidates else None
