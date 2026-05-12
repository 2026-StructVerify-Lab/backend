"""
sv_platform/alembic/env.py — Alembic 환경 설정

기능:
  - 환경변수 POSTGRES_DSN으로 sqlalchemy.url override (alembic.ini 값보다 우선)
  - asyncpg DSN을 동기 드라이버로 변환 (alembic은 sync)
  - target_metadata에 sv_platform.models.Base.metadata 연결 → autogenerate 가능
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── 프로젝트 루트를 sys.path에 추가 (alembic이 sv_platform.models import 가능하게)
HERE = Path(__file__).resolve()
BACKEND_ROOT = HERE.parent.parent.parent      # backend/
sys.path.insert(0, str(BACKEND_ROOT))

# ── sv_platform 모델 metadata 로드
from sv_platform.models import Base  # noqa: E402

target_metadata = Base.metadata

# ── Alembic Config object
config = context.config

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _get_url() -> str:
    """
    DB URL 결정 우선순위:
      1. 환경변수 POSTGRES_DSN
      2. alembic.ini의 sqlalchemy.url

    asyncpg DSN("postgresql+asyncpg://...")이면 동기 드라이버로 변환.
    """
    url = os.environ.get("POSTGRES_DSN") or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("POSTGRES_DSN not set")
    # alembic은 sync — asyncpg 드라이버 prefix 제거
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")
    return url


def run_migrations_offline() -> None:
    """오프라인 모드 — SQL만 출력."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """온라인 모드 — 실제 DB에 적용."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
