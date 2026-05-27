"""Component eval: catalog retrieval recall@k."""
from __future__ import annotations

from typing import Any

from structverify.eval.schemas import ComponentRetrievalRow
from structverify.retrieval.base_connector import ConnectorQuery
from structverify.retrieval.kosis_connector import KOSISConnector


async def run_retrieval_suite(
    rows: list[ComponentRetrievalRow],
    config: dict[str, Any],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    base_kosis = config.get("kosis") if isinstance(config.get("kosis"), dict) else {}
    llm_cfg = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    kosis_cfg = {**base_kosis}
    if llm_cfg:
        kosis_cfg["llm"] = llm_cfg
    connector = KOSISConnector(config=kosis_cfg)
    results: list[dict[str, Any]] = []
    for row in rows:
        query = ConnectorQuery(
            keyword=row.keyword,
            indicator=row.indicator or row.keyword,
            time_period=row.time_period,
            population=row.population,
        )
        try:
            candidates = await connector.catalog.search(query, top_k=max(k, 10))
            ids = [c.stat_id for c in candidates[:k]]
            correct = row.gold_stat_id in ids
            results.append(
                {
                    "row_id": row.row_id,
                    "correct": correct,
                    "gold_stat_id": row.gold_stat_id,
                    "top_k_ids": ids,
                }
            )
        except Exception as e:
            results.append(
                {"row_id": row.row_id, "correct": False, "error": str(e)}
            )
    return results
