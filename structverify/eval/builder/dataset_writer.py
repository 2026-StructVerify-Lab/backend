"""JSONL + manifest persistence for eval datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from structverify.eval.builder.schemas import BuildState, EvalArticle, EvalManifest
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetWriter:
    def __init__(self, output_dir: Path, dataset_id: str):
        self.dataset_dir = output_dir / dataset_id
        self.articles_path = self.dataset_dir / "articles.jsonl"
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.dataset_id = dataset_id

    def ensure_dirs(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def append_article(self, article: EvalArticle) -> None:
        self.ensure_dirs()
        line = article.model_dump(mode="json")
        with open(self.articles_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def load_articles(self) -> list[EvalArticle]:
        if not self.articles_path.exists():
            return []
        articles: list[EvalArticle] = []
        with open(self.articles_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                articles.append(EvalArticle.model_validate_json(line))
        return articles

    @staticmethod
    def sha256_file(path: Path) -> str | None:
        if not path.exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def config_hash(config: dict[str, Any]) -> str:
        blob = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def write_manifest(
        self,
        manifest: EvalManifest,
        config: dict[str, Any],
        status: str = "frozen",
    ) -> None:
        self.ensure_dirs()
        manifest.status = status  # type: ignore[assignment]
        manifest.articles_sha256 = self.sha256_file(self.articles_path)
        manifest.builder_config_hash = self.config_hash(config)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        logger.info(f"Manifest written: {self.manifest_path}")

    def write_build_state(self, state_path: Path, state: BuildState) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_build_state(state_path: Path, dataset_id: str, mode: str, seed: int) -> BuildState:
        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                data = json.load(f)
            return BuildState.model_validate(data)
        return BuildState(dataset_id=dataset_id, mode=mode, seed=seed)
