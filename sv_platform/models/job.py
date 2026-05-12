"""
sv_platform.models.job — 검증 Job

라이프사이클:
  pending → running → completed | failed

Phase 1 (이번 단계)에서는 sync로 실행되므로 거의 즉시 completed.
Phase 2부터 ARQ 워커로 비동기 실행.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, Uuid, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from sv_platform.models.base import Base, IdMixin, TimestampMixin


class Job(Base, IdMixin, TimestampMixin):
    __tablename__ = "jobs"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 어느 API 키로 제출됐는지 (UI 세션이면 NULL)
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        index=True,
    )

    # 입력 종류와 데이터
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)   # text/pdf/docx/url
    source_data: Mapped[str | None] = mapped_column(Text, nullable=True)   # 텍스트 입력
    source_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)  # 업로드 파일

    # 검증에 사용할 datasource id 목록 (default ["kosis"])
    datasources: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    callback_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_step: Mapped[str | None] = mapped_column(String(100), nullable=True)

    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<Job {self.id} status={self.status} source_type={self.source_type}>"
