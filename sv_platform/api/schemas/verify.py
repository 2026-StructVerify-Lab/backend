"""
sv_platform.api.schemas.verify — 검증 요청/응답 + Job 응답 스키마

프론트엔드 lib/types.ts의 ClaimResult / Job 모양과 정확히 일치해야 함.
변경 시 양쪽 동시 업데이트 필수.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ── 요청 ────────────────────────────────────────────────────────
SourceType = Literal["text", "pdf", "docx", "url"]


class VerifyRequest(BaseModel):
    """JSON body (text/url)용. 파일 업로드는 multipart로 별도 처리."""
    source_type: SourceType
    source_data: str | None = None        # text 입력 시
    url: str | None = None                 # url 입력 시
    datasources: list[str] | None = Field(default_factory=lambda: ["kosis"])
    callback_url: str | None = None
    options: dict[str, Any] | None = None


# ── Job 응답 ─────────────────────────────────────────────────────
class JobOut(BaseModel):
    id: UUID
    status: Literal["pending", "running", "completed", "failed"]
    source_type: SourceType
    source_uri: str | None = None
    source_data: str | None = None
    progress: int
    current_step: str | None
    result: dict | None                   # VerificationReport 전체 JSON
    error: str | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class JobSubmitResponse(BaseModel):
    """POST /v1/verify 즉시 응답 — job 생성됐다는 신호."""
    job_id: UUID
    status: str
    poll_url: str
    created_at: datetime
