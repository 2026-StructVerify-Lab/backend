"""Outcome eval harness runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from structverify.core.config_loader import load_config
from structverify.eval.io import (
    append_jsonl,
    ensure_run_dir,
    filter_cases_by_split,
    load_models,
    load_outcome_manifest,
    load_yaml,
    make_run_id,
    write_json,
)
from structverify.eval.outcome.runtime_slice import run_outcome_slice
from structverify.eval.outcome.scorer import score_case
from structverify.eval.outcome.sir_factory import build_sir_for_case, claims_from_case
from structverify.eval.report import summarize_outcome_nested
from structverify.eval.schemas import OutcomeCase
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def merge_eval_config(harness_cfg: dict[str, Any]) -> dict[str, Any]:
    app = load_config()
    merged = {**app, **harness_cfg}
    merged["eval"] = {**(app.get("eval") or {}), **(harness_cfg.get("eval") or {})}
    app_kosis = app.get("kosis") if isinstance(app.get("kosis"), dict) else {}
    harness_kosis = harness_cfg.get("kosis") if isinstance(harness_cfg.get("kosis"), dict) else {}
    kosis = {**app_kosis, **harness_kosis}
    llm_cfg = app.get("llm") if isinstance(app.get("llm"), dict) else {}
    if llm_cfg:
        kosis["llm"] = llm_cfg
    merged["kosis"] = kosis
    return merged


class OutcomeEvalRunner:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        datasets_root: Path = Path("eval/datasets"),
        runs_root: Path = Path("eval/runs"),
        split: str | None = None,
    ):
        self.config = merge_eval_config(config)
        self.datasets_root = Path(datasets_root)
        self.runs_root = Path(runs_root)
        self.dataset_id = config.get("dataset_id", "structverify_outcome_v1")
        eval_cfg = self.config.get("eval") or {}
        self.split = split or eval_cfg.get("split", "train")

    def load_cases(self, *, limit: int | None = None) -> list[OutcomeCase]:
        path = self.datasets_root / self.dataset_id / "claims.jsonl"
        cases = load_models(path, OutcomeCase)
        manifest = load_outcome_manifest(self.datasets_root, self.dataset_id)
        holdout_ids = set(manifest.holdout_case_ids) if manifest else None
        cases = filter_cases_by_split(
            cases, holdout_ids=holdout_ids, split=self.split
        )
        return cases[:limit] if limit else cases

    async def run(
        self,
        *,
        limit: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        cases = self.load_cases(limit=limit)
        rid = run_id or make_run_id(self.dataset_id, prefix="outcome")
        out_dir = ensure_run_dir(self.runs_root, rid)
        pred_path = out_dir / "predictions.jsonl"
        if pred_path.exists():
            pred_path.unlink()

        eval_cfg = self.config.get("eval") or {}
        rel = float(eval_cfg.get("value_tolerance_relative", 0.005))
        abs_tol = float(eval_cfg.get("value_tolerance_absolute", 0.1))
        schema_modes: list[str] = list(
            eval_cfg.get("schema_modes") or ["induce"]
        )
        primary_mode = str(
            eval_cfg.get("primary_schema_mode") or schema_modes[0]
        )

        predictions: list[dict[str, Any]] = []

        for case in cases:
            for schema_mode in schema_modes:
                logger.info(
                    f"Outcome eval {case.case_id} schema_mode={schema_mode}"
                )
                cfg = dict(self.config)
                if case.domain and eval_cfg.get("domain_oracle", True):
                    cfg["detected_domain"] = case.domain
                    cfg["detected_domain_desc"] = case.domain
                try:
                    sir_doc = build_sir_for_case(case)
                    oracle = claims_from_case(
                        case, sir_doc, schema_mode=schema_mode
                    )
                    _, results = await run_outcome_slice(
                        sir_doc,
                        oracle,
                        cfg,
                        schema_mode=schema_mode,
                        case_id=case.case_id,
                    )
                    result = results[0] if results else None
                    rec = score_case(
                        case,
                        result,
                        rel=rel,
                        abs_tol=abs_tol,
                        schema_mode=schema_mode,
                    )
                except Exception as e:
                    logger.exception(
                        f"Outcome case failed {case.case_id} "
                        f"mode={schema_mode}: {e}"
                    )
                    rec = score_case(
                        case,
                        None,
                        error=str(e),
                        schema_mode=schema_mode,
                    )
                predictions.append(rec.model_dump())
                append_jsonl(pred_path, rec.model_dump())

        outcome_summary = summarize_outcome_nested(
            predictions, primary_schema_mode=primary_mode
        )
        report = {
            "run_id": rid,
            "dataset_id": self.dataset_id,
            "axis": "outcome",
            "split": self.split,
            "outcome": outcome_summary,
            "predictions_path": str(pred_path),
        }
        write_json(out_dir / "report.json", report)
        write_json(
            out_dir / "run_meta.json",
            {
                "run_id": rid,
                "dataset_id": self.dataset_id,
                "limit": limit,
                "split": self.split,
                "schema_modes": schema_modes,
            },
        )
        primary_block = outcome_summary.get(primary_mode) or {}
        logger.info(
            f"Outcome eval done [{primary_mode}]: "
            f"verdict_accuracy={primary_block.get('verdict_accuracy', 0):.3f}"
        )
        return report
