"""JSONL I/O and eval run directory helpers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any] | BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                f.write(row.model_dump_json() + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any] | BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if isinstance(row, BaseModel):
            f.write(row.model_dump_json() + "\n")
        else:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_models(path: Path, model_cls: type[T]) -> list[T]:
    return [model_cls.model_validate(r) for r in read_jsonl(path)]


def iter_models(path: Path, model_cls: type[T]) -> Iterator[T]:
    for row in read_jsonl(path):
        yield model_cls.model_validate(row)


def dataset_dir(datasets_root: Path, dataset_id: str) -> Path:
    return datasets_root / dataset_id


def make_run_id(dataset_id: str, prefix: str = "eval") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{dataset_id}_{ts}"


def ensure_run_dir(runs_root: Path, run_id: str) -> Path:
    out = runs_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
