"""EvalHarness — run VerificationPipeline on golden set and collect metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from structverify.core.config_loader import load_config
from structverify.core.pipeline import VerificationPipeline
from structverify.eval.builder.dataset_writer import DatasetWriter
from structverify.eval.builder.schemas import EvalArticle, EvalManifest
from structverify.eval.harness.loader import load_golden_dataset
from structverify.eval.harness.matcher import (
    article_prediction_record,
    match_golden_claims,
)
from structverify.eval.harness.metrics import compute_report
from structverify.eval.harness.report_io import (
    append_prediction,
    ensure_run_dir,
    load_predictions,
    make_run_id,
    run_dir,
    write_report,
    write_run_meta,
)
from structverify.eval.harness.visualize import render_charts_and_html
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def load_eval_harness_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_harness_config(harness_cfg: dict[str, Any]) -> dict[str, Any]:
    app_cfg = load_config()
    merged = {**app_cfg, **harness_cfg}
    merged["persist_to_db"] = bool(harness_cfg.get("persist_to_db", False))
    merged["enable_feedback"] = bool(harness_cfg.get("enable_feedback", False))
    cd: dict[str, Any] = {}
    cd.update(app_cfg.get("candidate_detection") or {})
    cd.update(harness_cfg.get("candidate_detection") or {})
    if cd:
        merged["candidate_detection"] = cd
    return merged


class EvalHarness:
    """Evaluate pipeline against a frozen golden dataset."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        datasets_root: str | Path = "eval/datasets",
        dataset_id: str = "structverify_eval_v1",
        runs_dir: str | Path = "eval/runs",
    ):
        self.config = config or {}
        self.datasets_root = Path(datasets_root)
        self.dataset_id = dataset_id
        self.runs_dir = Path(runs_dir)
        self._manifest: EvalManifest | None = None
        self._articles: list[EvalArticle] = []

    def load(self) -> None:
        self._manifest, self._articles = load_golden_dataset(
            self.datasets_root, self.dataset_id
        )
        logger.info(
            f"Loaded eval set {self.dataset_id}: "
            f"{len(self._articles)} articles, status={self._manifest.status}"
        )

    @property
    def manifest(self) -> EvalManifest:
        if self._manifest is None:
            self.load()
        assert self._manifest is not None
        return self._manifest

    @property
    def articles(self) -> list[EvalArticle]:
        if not self._articles:
            self.load()
        return self._articles

    async def run(
        self,
        *,
        limit: int | None = None,
        run_id: str | None = None,
        skip_viz: bool = False,
        resume: bool = False,
        existing_run_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run pipeline on each golden article; write predictions + report + charts."""
        self.load()
        articles = self.articles[:limit] if limit else self.articles

        if existing_run_dir is not None:
            out = Path(existing_run_dir)
            rid = out.name
        else:
            rid = run_id or make_run_id(self.dataset_id)
            out = ensure_run_dir(run_dir(self.runs_dir, rid))

        pred_path = out / "predictions.jsonl"
        article_records: list[dict[str, Any]] = []

        if resume and pred_path.exists():
            article_records = load_predictions(pred_path)
            done_ids = {r["article_id"] for r in article_records}
            articles = [a for a in articles if a.article_id not in done_ids]
            logger.info(
                f"Resume: {len(done_ids)} articles done, {len(articles)} remaining"
            )

        pipeline = VerificationPipeline(self.config)
        writer = DatasetWriter(self.datasets_root, self.dataset_id)
        manifest_sha = writer.sha256_file(writer.articles_path)

        for article in articles:
            logger.info(f"Eval article {article.article_id}")
            try:
                report = await pipeline.run(article.article_text, "text")
                rows, extra = match_golden_claims(
                    article.claims, report.claims, report.results
                )
                rec = article_prediction_record(
                    article.article_id,
                    article.intended_domain,
                    getattr(article, "article_scope", "local") or "local",
                    report.document.detected_domain,
                    rows,
                    extra,
                    len(report.claims),
                    len(report.results),
                )
            except Exception as e:
                logger.exception(f"Pipeline failed for {article.article_id}: {e}")
                rec = article_prediction_record(
                    article.article_id,
                    article.intended_domain,
                    getattr(article, "article_scope", "local") or "local",
                    None,
                    [],
                    0,
                    0,
                    0,
                    error=str(e),
                )
            article_records.append(rec)
            append_prediction(pred_path, rec)

        full_report = compute_report(
            self.dataset_id,
            rid,
            article_records,
            manifest_sha256=manifest_sha,
        )
        write_report(out / "report.json", full_report)
        write_run_meta(
            out / "run_meta.json",
            {
                "run_id": rid,
                "dataset_id": self.dataset_id,
                "article_limit": limit,
                "persist_to_db": self.config.get("persist_to_db"),
            },
        )

        if not skip_viz:
            render_charts_and_html(out, full_report)

        logger.info(
            f"Eval run complete: {out} "
            f"verdict_accuracy={full_report['summary']['verdict_accuracy']:.3f}"
        )
        return full_report
