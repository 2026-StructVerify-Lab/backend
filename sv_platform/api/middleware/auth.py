"""
sv_platform.api.middleware.auth — Bearer 헤더 인증 dependency

두 가지 인증 방식 동시 지원:
  1. JWT (UI 세션) — payload에 user_id + tenant_id
  2. API key (외부 SDK) — DB lookup으로 tenant_id 해석

같은 endpoint가 둘 다 받음. 호출 측은 AuthContext를 받아 tenant_id로 데이터 격리.

[사용]
    from sv_platform.api.middleware.auth import get_auth, AuthContext

    @router.post("/v1/verify")
    async def verify(
        req: VerifyRequest,
        ctx: AuthContext = Depends(get_auth),
        db: AsyncSession = Depends(get_session),
    ):
        # ctx.tenant_id 가용
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from sv_platform.auth.api_key import verify_api_key
from sv_platform.auth.jwt_handler import TokenError, decode_access_token
from sv_platform.db import get_session


@dataclass
class AuthContext:
    """인증된 요청의 컨텍스트. tenant_id는 항상 있음."""
    tenant_id: UUID
    user_id: Optional[UUID]            # JWT일 때만 채워짐
    api_key_id: Optional[UUID]          # API key일 때만 채워짐
    auth_type: str                      # "jwt" | "api_key"


async def get_auth(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_session),
) -> AuthContext:
    """
    Bearer 토큰을 검사해서 AuthContext 반환.

    - "Bearer sv_..."  → API key 경로 (DB lookup)
    - "Bearer <기타>"   → JWT 경로

    실패 시 401 반환.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization must start with 'Bearer '",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[len("Bearer ") :].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty token",
        )

    # ── 분기: prefix로 결정 ─────────────────────────────────────
    if token.startswith("sv_"):
        # API key
        key = await verify_api_key(token, db)
        if key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key",
            )
        # last_used_at 업데이트는 verify_api_key 안에서 flush됨
        # commit은 라우트 핸들러에서 한 번에 하므로 여기선 안 함
        return AuthContext(
            tenant_id=key.tenant_id,
            user_id=None,
            api_key_id=key.id,
            auth_type="api_key",
        )
    else:
        # JWT
        try:
            payload = decode_access_token(token)
        except TokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid JWT: {e}",
            )
        try:
            return AuthContext(
                tenant_id=UUID(payload["tenant_id"]),
                user_id=UUID(payload["sub"]),
                api_key_id=None,
                auth_type="jwt",
            )
        except (KeyError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT payload malformed",
            )
