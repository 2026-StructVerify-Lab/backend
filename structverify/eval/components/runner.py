"""Run component eval suites."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from structverify.eval.components.detection_runner import run_detection_suite
from structverify.eval.components.retrieval_runner import run_retrieval_suite
from structverify.eval.components.schema_runner import run_schema_suite
from structverify.eval.components.verdict_runner import run_verdict_suite
from structverify.eval.io import load_models, write_json
from structverify.eval.report import (
    summarize_component_suite,
    summarize_detection,
    summarize_schema_suite,
    summarize_verdict_suite,
)
from structverify.eval.schemas import (
    ComponentDetectionRow,
    ComponentRetrievalRow,
    ComponentSchemaRow,
    ComponentVerdictRow,
)
from structverify.eval.outcome.runner import merge_eval_config


class ComponentEvalRunner:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        datasets_root: Path = Path("eval/datasets"),
    ):
        self.config = merge_eval_config(config)
        self.datasets_root = Path(datasets_root)
        self.dataset_id = config.get(
            "component_dataset_id", "structverify_components_v1"
        )

    def _path(self, name: str) -> Path:
        return self.datasets_root / self.dataset_id / name

    async def run_suite(self, suite: str) -> dict[str, Any]:
        base = self.datasets_root / self.dataset_id
        if suite == "detection":
            rows = load_models(base / "detection.jsonl", ComponentDetectionRow)
            raw = await run_detection_suite(rows, self.config)
            return summarize_detection(raw)
        if suite == "schema":
            rows = load_models(base / "schema.jsonl", ComponentSchemaRow)
            raw = await run_schema_suite(rows, self.config)
            return summarize_schema_suite(raw)
        if suite == "retrieval":
            rows = load_models(base / "retrieval.jsonl", ComponentRetrievalRow)
            raw = await run_retrieval_suite(rows, self.config)
            return summarize_component_suite(raw)
        if suite == "verdict":
            rows = load_models(base / "verdict.jsonl", ComponentVerdictRow)
            raw = await run_verdict_suite(rows, self.config)
            return summarize_verdict_suite(raw)
        raise ValueError(f"Unknown suite: {suite}")

    async def run_all(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for suite in ("detection", "schema", "retrieval", "verdict"):
            p = self._path(f"{suite}.jsonl")
            if p.exists():
                out[suite] = await self.run_suite(suite)
        return out
