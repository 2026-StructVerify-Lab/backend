"""
sv_platform.models.api_key — API 키

- raw key는 DB에 *저장 안 함*. 발급 시 1회만 응답 반환.
- prefix는 인덱스용 (검색 빠르게), 사용자에게 식별 표시용 (sv_live_a1b2... 같이).
- hash는 argon2.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Uuid, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from sv_platform.models.base import Base, IdMixin, TimestampMixin


class ApiKey(Base, IdMixin, TimestampMixin):
    __tablename__ = "api_keys"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # "sv_live_a1b2c3" 형태 — 앞 12자. 검증 시 candidates 좁히는 용도.
    prefix: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # argon2 hash of full raw key
    hash: Mapped[str] = mapped_column(String(255), nullable=False)

    name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # ["verify:write", "datasources:read", ...]
    scopes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=60, nullable=False)

    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<ApiKey {self.id} prefix={self.prefix!r} name={self.name!r}>"