"""
sv_platform.api.routes.auth — /v1/auth/* 라우트

회원가입 (tenant + user 동시 생성) / 로그인 (비밀번호 검증 + JWT 발급).
SaaS UI에서 사용하는 경로. API key 인증은 별도 endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sv_platform.api.middleware.auth import AuthContext, get_auth
from sv_platform.api.schemas.auth import (
    AuthResponse,
    LoginRequest,
    SignupRequest,
    TenantOut,
    UserOut,
)
from sv_platform.auth.jwt_handler import create_access_token
from sv_platform.auth.password import hash_password, verify_password
from sv_platform.db import get_session
from sv_platform.models.tenant import Tenant
from sv_platform.models.user import User


router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    req: SignupRequest,
    db: AsyncSession = Depends(get_session),
) -> AuthResponse:
    """
    새 tenant + 새 user 동시 생성. user는 자동으로 owner 권한.

    이미 같은 email이 있으면 409.
    """
    # email 중복 체크
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # tenant 생성
    tenant = Tenant(name=req.tenant_name, plan="free")
    db.add(tenant)
    await db.flush()       # tenant.id 채워짐

    # user 생성
    user = User(
        tenant_id=tenant.id,
        email=req.email,
        password_hash=hash_password(req.password),
        role="owner",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    await db.refresh(tenant)

    token = create_access_token(user_id=user.id, tenant_id=tenant.id)
    return AuthResponse(
        access_token=token,
        user=UserOut.model_validate(user),
        tenant=TenantOut.model_validate(tenant),
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_session),
) -> AuthResponse:
    """
    이메일 + 비밀번호 검증 후 JWT 반환.

    실패 시 일관되게 401 (이메일 존재/비번 오류 구분 안 함 — 정보 누설 방지).
    """
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # tenant도 함께 응답에 포함
    tenant = await db.get(Tenant, user.tenant_id)
    if tenant is None:
        # 정합성 깨진 케이스 — 거의 불가능
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant not found",
        )

    token = create_access_token(user_id=user.id, tenant_id=tenant.id)
    return AuthResponse(
        access_token=token,
        user=UserOut.model_validate(user),
        tenant=TenantOut.model_validate(tenant),
    )


# ── 현재 사용자 정보 조회 ──────────────────────────────────────
@router.get("/me", response_model=AuthResponse)
async def get_me(
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
) -> AuthResponse:
    """
    현재 토큰의 사용자 + 테넌트 정보 반환.

    프론트엔드가 페이지 로드 시 호출해서 사이드바에 표시.
    JWT 인증만 지원 (API key로는 user 정보가 없으므로 400).
    """
    if ctx.auth_type != "jwt" or ctx.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint requires UI session (JWT)",
        )

    user = await db.get(User, ctx.user_id)
    tenant = await db.get(Tenant, ctx.tenant_id)
    if user is None or tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User or tenant not found",
        )

    # access_token은 같은 걸 다시 보내줄 필요는 없지만 schema 호환을 위해 빈 문자열.
    # 또는 새로 발급 — 여기선 빈 문자열로 (프론트가 이미 가지고 있음).
    return AuthResponse(
        access_token="",
        user=UserOut.model_validate(user),
        tenant=TenantOut.model_validate(tenant),
    )