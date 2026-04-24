from __future__ import annotations
import os
from pathlib import Path

def load_config(path: str | None = None) -> dict:
    # .env 파일 자동 로드
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    import yaml
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "config" / "default.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}