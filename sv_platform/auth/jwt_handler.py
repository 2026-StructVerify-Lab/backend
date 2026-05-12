"""
sv_platform.auth.jwt_handler — JWT 발급/검증 (UI 세션용)

SaaS UI에서 로그인하면 JWT 발급, 이후 요청에 Bearer 헤더로 첨부.
API key와는 다른 인증 방식이지만 같은 middleware에서 둘 다 처리.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from jose import JWTError, jwt

from sv_platform.config import settings


class TokenError(Exception):
    """JWT 검증 실패 — 만료, 변조, 형식 오류 등."""
    pass


def create_access_token(
    user_id: UUID,
    tenant_id: UUID,
    ttl_minutes: int | None = None,
) -> str:
    """
    사용자 로그인 직후 발급. payload 최소화:
      - sub        : user_id
      - tenant_id  : 멀티 테넌시 분리용
      - exp        : 만료 시각

    Returns: encoded JWT 문자열
    """
    minutes = ttl_minutes or settings.auth.jwt_access_token_ttl_minutes
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(
        payload,
        settings.auth.jwt_secret,
        algorithm=settings.auth.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict:
    """
    JWT 검증 + 디코딩. 실패 시 TokenError.

    Returns: payload dict ({"sub", "tenant_id", "iat", "exp"})
    """
    try:
        payload = jwt.decode(
            token,
            settings.auth.jwt_secret,
            algorithms=[settings.auth.jwt_algorithm],
        )
        return payload
    except JWTError as e:
        raise TokenError(str(e))
