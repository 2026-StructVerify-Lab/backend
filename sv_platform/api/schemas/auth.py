"""
sv_platform.api.schemas.auth — 인증 요청/응답 Pydantic 스키마
"""
from __future__ import annotations

from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ── 회원가입 ────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    tenant_name: str = Field(min_length=1, max_length=100)


# ── 로그인 ──────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ── 응답 ────────────────────────────────────────────────────────
class UserOut(BaseModel):
    id: UUID
    email: str
    role: str
    tenant_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class TenantOut(BaseModel):
    id: UUID
    name: str
    plan: str

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    """signup/login 공통 응답."""
    access_token: str
    token_type: str = "bearer"
    user: UserOut
    tenant: TenantOut
