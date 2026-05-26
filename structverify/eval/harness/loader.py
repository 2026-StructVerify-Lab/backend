"""Load frozen golden eval datasets from eval/datasets/."""
from __future__ import annotations

import json
from pathlib import Path

from structverify.eval.builder.dataset_writer import DatasetWriter
from structverify.eval.builder.schemas import EvalArticle, EvalManifest


def load_manifest(dataset_dir: Path) -> EvalManifest:
    path = dataset_dir / "manifest.json"
    with open(path, encoding="utf-8") as f:
        return EvalManifest.model_validate(json.load(f))


def load_golden_dataset(
    datasets_root: Path | str,
    dataset_id: str,
) -> tuple[EvalManifest, list[EvalArticle]]:
    """Load manifest + articles for harness evaluation."""
    writer = DatasetWriter(Path(datasets_root), dataset_id)
    manifest = load_manifest(writer.dataset_dir)
    articles = writer.load_articles()
    return manifest, articles
