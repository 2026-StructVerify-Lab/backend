"""
sv_platform.models.user — SaaS UI 로그인 사용자

API key는 user가 아닌 *tenant*에 직접 묶임 (서비스 계정 개념).
user는 UI 인증(JWT)을 위한 존재.
"""
from __future__ import annotations

import uuid

from sqlalchemy import String, Uuid, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sv_platform.models.base import Base, IdMixin, TimestampMixin
from sv_platform.models.tenant import Tenant


class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "users"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20),
        default="owner",
        nullable=False,
        # owner / admin / member — 같은 tenant 안에서 권한 분리
    )

    # 관계 (lazy load, async에서 explicit하게 selectinload 권장)
    tenant: Mapped[Tenant] = relationship(lazy="raise")

    def __repr__(self) -> str:
        return f"<User {self.id} email={self.email!r} role={self.role!r}>"
