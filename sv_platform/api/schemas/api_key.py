"""
sv_platform.api.schemas.api_key — API 키 발급/조회 Pydantic 스키마
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] | None = None
    rate_limit_per_min: int | None = 60


class ApiKeyOut(BaseModel):
    """일반 조회 응답 — raw key 안 포함."""
    id: UUID
    prefix: str
    name: str | None
    scopes: list[str] | None
    rate_limit_per_min: int
    last_used_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class ApiKeyCreateResponse(BaseModel):
    """발급 직후 응답 — raw `key`가 *유일하게 보이는 시점*."""
    key: str                          # raw API key (1회만)
    api_key: ApiKeyOut
