"""
sv_platform.models.base — SQLAlchemy 공통 베이스

모든 ORM 모델은 여기 Base를 상속.
Alembic env.py가 Base.metadata로 마이그레이션 생성.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """기본 timestamp 생성 — timezone-aware UTC."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """모든 ORM 모델의 베이스."""
    pass


class IdMixin:
    """UUID primary key — 모든 테이블 공통."""
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """created_at / updated_at — 거의 모든 테이블 공통."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )
