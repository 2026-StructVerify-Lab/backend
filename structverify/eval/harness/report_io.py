"""Read/write eval harness run artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_run_id(dataset_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{dataset_id}_{ts}"


def run_dir(runs_root: Path, run_id: str) -> Path:
    return runs_root / run_id


def ensure_run_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "charts").mkdir(exist_ok=True)
    return path


def append_prediction(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_report(path: Path, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def load_report(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_run_meta(path: Path, meta: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
