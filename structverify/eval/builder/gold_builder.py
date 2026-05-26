"""Build gold labels from KOSIS fetch probe (code-owned, not LLM)."""
from __future__ import annotations

import random
import re
from typing import Any

import httpx

from structverify.eval.builder.schemas import ClaimSpec, GoldEvidence, GoldSchema, MismatchRecipe
from structverify.eval.builder.text_utils import normalize_kosis_unit
from structverify.retrieval.base_connector import ConnectorQuery, StatRecord
from structverify.retrieval.kosis_connector import (
    KOSISConnector,
    _rows_from_kosis_body,
    kosis_enrich_stat_records,
)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _latest_year_from_stat_rec(stat_rec: StatRecord) -> str | None:
    """getMeta PRD에서 사용 가능한 최신 연도 추출 (2024 하드코드 fetch 방지)."""
    prd = (stat_rec.metadata or {}).get("getMeta_PRD")
    rows = _rows_from_kosis_body(prd) if prd else []
    years: list[int] = []
    for row in rows:
        prd_de = row.get("PRD_DE")
        if prd_de is None:
            continue
        m = re.search(r"(\d{4})", str(prd_de))
        if m:
            years.append(int(m.group(1)))
    if years:
        return str(max(years))
    return None


def _indicator_from_stat(stat_name: str, raw: dict[str, Any] | None = None) -> str:
    if raw:
        for key in ("ITM_NM", "C1_NM", "C2_NM"):
            val = raw.get(key)
            if val and str(val).strip():
                return str(val).strip()
    return stat_name.split("(")[0].strip() or stat_name


def _perturb_value(official: float, rng: random.Random) -> float:
    factor = rng.choice([1.15, 1.25, 0.75, 0.85, 1.3])
    perturbed = official * factor
    if abs(perturbed - official) < 0.01:
        perturbed = official + rng.choice([1.0, -1.0, 5.0, -5.0])
    return round(perturbed, 4)


def _perturb_time_period(period: str, rng: random.Random) -> str:
    m = re.match(r"^(\d{4})(.*)$", period or "")
    if not m:
        return period
    year = int(m.group(1)) + rng.choice([-1, 1])
    return f"{year}{m.group(2)}"


class GoldBuilder:
    """KOSIS-first gold claim spec builder."""

    def __init__(self, config: dict | None = None, seed: int = 42):
        self.config = config or {}
        self.rng = random.Random(seed)
        kosis_cfg = {**self.config.get("kosis", {}), **self.config}
        self.connector = KOSISConnector(config=kosis_cfg)
        self.timeout = float(self.config.get("kosis", {}).get("fetch_probe_timeout_sec", 30))
        self.probe_cache_enabled = bool(
            kosis_cfg.get("probe_cache_enabled", True)
        )
        self._probe_cache: dict[str, dict[str, Any] | None] = {}
        self._enriched_stat_ids: set[str] = set()
        self._stat_metadata_cache: dict[str, dict[str, Any]] = {}

    def clear_probe_cache(self) -> None:
        """빌드 세션 종료 시 probe 캐시 비우기 (기사마다 호출하지 않음)."""
        self._probe_cache.clear()

    def _stat_record_from_row(self, row: dict[str, Any]) -> StatRecord:
        keywords = row.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        return StatRecord(
            stat_id=row["stat_id"],
            stat_name=row.get("stat_name") or row["stat_id"],
            org_id=row.get("org_id"),
            org_name=row.get("org_name") or "",
            metadata={
                "category_path": row.get("category_path"),
                "keywords": keywords,
            },
        )

    async def _enrich_stat_record(self, stat_rec: StatRecord) -> None:
        base = (self.connector.config.get("base_url") or KOSISConnector.BASE_URL).rstrip("/")
        api_key = self.connector.api_key
        if not api_key or not stat_rec.org_id:
            return
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await kosis_enrich_stat_records(
                    client,
                    base,
                    api_key,
                    [stat_rec],
                    timeout=self.timeout,
                )
        except Exception as e:
            logger.debug(f"getMeta enrich skipped: {e}")

    async def fetch_probe(self, catalog_row: dict[str, Any]) -> dict[str, Any] | None:
        stat_id = str(catalog_row.get("stat_id") or "")
        if self.probe_cache_enabled and stat_id and stat_id in self._probe_cache:
            cached = self._probe_cache[stat_id]
            if cached is None:
                return None
            return {**cached, "catalog_row": catalog_row}

        stat_rec = self._stat_record_from_row(catalog_row)
        if stat_id and stat_id in self._stat_metadata_cache:
            stat_rec.metadata.update(self._stat_metadata_cache[stat_id])
        elif stat_id not in self._enriched_stat_ids:
            await self._enrich_stat_record(stat_rec)
            if stat_id:
                self._enriched_stat_ids.add(stat_id)
                self._stat_metadata_cache[stat_id] = dict(stat_rec.metadata)

        hint_year = _latest_year_from_stat_rec(stat_rec) or "2023"

        query = ConnectorQuery(
            keyword=stat_rec.stat_name,
            indicator=stat_rec.stat_name,
            time_period=hint_year,
        )
        data = await self.connector.fetch(
            stat_id=stat_rec.stat_id,
            params={"stat_record": stat_rec, "query": query},
        )
        if data is None or data.official_value is None:
            logger.debug(
                f"fetch probe failed: {stat_rec.stat_id} (hint_year={hint_year})"
            )
            if self.probe_cache_enabled and stat_id:
                self._probe_cache[stat_id] = None
            return None

        raw_row = (data.raw_response or {}).get("row") or {}
        indicator = _indicator_from_stat(
            stat_rec.stat_name, raw_row if isinstance(raw_row, dict) else None
        )
        probed = {
            "stat_id": stat_rec.stat_id,
            "stat_name": stat_rec.stat_name,
            "org_name": stat_rec.org_name,
            "category_path": catalog_row.get("category_path")
            or stat_rec.metadata.get("category_path"),
            "official_value": float(data.official_value),
            "unit": data.unit,
            "time_period": data.time_period or "",
            "indicator": indicator,
        }
        if self.probe_cache_enabled and stat_id:
            self._probe_cache[stat_id] = probed
        return {**probed, "catalog_row": catalog_row}

    async def build_verifiable_spec(
        self,
        claim_id: str,
        catalog_row: dict[str, Any],
        verdict: str,
    ) -> ClaimSpec | None:
        probed = await self.fetch_probe(catalog_row)
        if probed is None:
            return None

        official = probed["official_value"]
        time_period = probed["time_period"] or "2023"
        unit = normalize_kosis_unit(probed.get("unit"))
        indicator = probed["indicator"]
        mismatch_recipe: MismatchRecipe | None = None
        claimed_value = official
        claimed_time = time_period

        if verdict == "mismatch":
            mismatch_recipe = self.rng.choice(["value", "time"])
            if mismatch_recipe == "value":
                claimed_value = _perturb_value(official, self.rng)
            else:
                claimed_time = _perturb_time_period(time_period, self.rng)

        gold_schema = GoldSchema(
            indicator=indicator,
            value=claimed_value,
            unit=unit,
            time_period=claimed_time,
            population="전체",
        )
        evidence = GoldEvidence(
            stat_name=probed["stat_name"],
            category_path=probed.get("category_path"),
            org_name=probed.get("org_name"),
        )
        return ClaimSpec(
            claim_id=claim_id,
            intended_verdict=verdict,  # type: ignore[arg-type]
            gold_schema=gold_schema,
            gold_stat_id=probed["stat_id"],
            gold_official_value=official,
            gold_evidence=evidence,
            mismatch_recipe=mismatch_recipe,
            catalog_row=catalog_row,
        )
