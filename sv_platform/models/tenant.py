"""
sv_platform.models.tenant — 테넌트 (회사/계정)

한 tenant 안에 여러 user / api_key / job / datasource가 속함.
"""
from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from sv_platform.models.base import Base, IdMixin, TimestampMixin


class Tenant(Base, IdMixin, TimestampMixin):
    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    plan: Mapped[str] = mapped_column(
        String(20),
        default="free",
        nullable=False,
        # free / pro / enterprise — 향후 billing 연동 시 사용
    )

    def __repr__(self) -> str:
        return f"<Tenant {self.id} name={self.name!r} plan={self.plan!r}>"
