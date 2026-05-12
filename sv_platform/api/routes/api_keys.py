"""
sv_platform.api.routes.api_keys — /v1/api-keys CRUD

UI 세션(JWT)로만 호출 — API 키로 API 키를 만들거나 취소하는 건 막음.
(보안: 노출된 키로 다른 키 발급 못 하게)
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sv_platform.api.middleware.auth import AuthContext, get_auth
from sv_platform.api.schemas.api_key import (
    ApiKeyCreate,
    ApiKeyCreateResponse,
    ApiKeyOut,
)
from sv_platform.auth.api_key import generate_api_key
from sv_platform.db import get_session
from sv_platform.models.api_key import ApiKey


router = APIRouter(prefix="/v1/api-keys", tags=["api_keys"])


def _ensure_jwt_auth(ctx: AuthContext) -> None:
    """JWT 세션에서만 키 관리 허용."""
    if ctx.auth_type != "jwt":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key management requires UI session (not API key auth)",
        )


@router.get("", response_model=list[ApiKeyOut])
async def list_keys(
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
) -> list[ApiKeyOut]:
    """현재 tenant의 모든 키 (revoke된 것 포함)."""
    _ensure_jwt_auth(ctx)
    stmt = (
        select(ApiKey)
        .where(ApiKey.tenant_id == ctx.tenant_id)
        .order_by(ApiKey.created_at.desc())
    )
    result = await db.execute(stmt)
    keys = result.scalars().all()
    return [ApiKeyOut.model_validate(k) for k in keys]


@router.post(
    "",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_key(
    req: ApiKeyCreate,
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
) -> ApiKeyCreateResponse:
    """
    새 API 키 발급.

    응답의 `key` 필드는 *raw API key*로, 이 시점 외에는 다시 볼 수 없음.
    DB에는 argon2 해시만 저장.
    """
    _ensure_jwt_auth(ctx)

    raw, prefix, key_hash = generate_api_key("live")

    api_key = ApiKey(
        tenant_id=ctx.tenant_id,
        prefix=prefix,
        hash=key_hash,
        name=req.name,
        scopes=req.scopes,
        rate_limit_per_min=req.rate_limit_per_min or 60,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return ApiKeyCreateResponse(
        key=raw,
        api_key=ApiKeyOut.model_validate(api_key),
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(
    key_id: UUID,
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
):
    """키 취소 (soft delete — revoked_at 설정)."""
    _ensure_jwt_auth(ctx)

    key = await db.get(ApiKey, key_id)
    if key is None or key.tenant_id != ctx.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    if key.revoked_at is not None:
        return  # 이미 취소됨 — idempotent

    key.revoked_at = datetime.now(timezone.utc)
    await db.commit()
    # 204 응답이라 body 없음 — return 안 함