"""
sv_platform.auth.api_key — API 키 생성/검증

[키 형식]
    sv_<env>_<random>
    예: sv_live_a1b2c3d4_8XlmNqRtZ7sV3pYK...

    - prefix (앞 12자): "sv_live_a1b2" 형태. DB에서 인덱스 검색용.
    - 전체: argon2 해시로 저장. raw 값은 발급 시 1회만 사용자에게 반환.

[보안]
    - argon2 hash + 환경변수 pepper (settings.auth.api_key_pepper)를 raw key 뒤에 append.
      DB가 유출돼도 pepper 모르면 hash 검증 불가능.
"""
import secrets
from datetime import datetime, timezone
from uuid import UUID

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sv_platform.config import settings
from sv_platform.models.api_key import ApiKey


_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# raw key 앞에 붙는 식별 prefix 길이 ("sv_live_a1b2" = 12)
_PREFIX_LEN = 12


def _pepper(raw: str) -> str:
    """pepper와 raw key 결합 (해시 입력으로 들어감)."""
    return raw + settings.auth.api_key_pepper


def generate_api_key(env: str = "live") -> tuple[str, str, str]:
    """
    새 API 키를 생성.

    Returns:
        (raw_key, prefix, hash)
        raw_key — 사용자에게 반환 (1회만 표시)
        prefix  — DB에 인덱스용으로 저장
        hash    — DB에 argon2 hash로 저장 (raw는 저장 안 함)

    Example:
        raw, prefix, h = generate_api_key("live")
        # raw    = "sv_live_a1b2c3d4_AbCdEf..."
        # prefix = "sv_live_a1b2"
        # h      = "$argon2id$..."
    """
    assert env in ("live", "test"), "env must be 'live' or 'test'"
    rand = secrets.token_urlsafe(32)
    raw = f"sv_{env}_{rand}"
    prefix = raw[:_PREFIX_LEN]
    h = _pwd_context.hash(_pepper(raw))
    return raw, prefix, h


async def verify_api_key(raw: str, db: AsyncSession) -> ApiKey | None:
    """
    raw key 검증. 유효하면 ApiKey 객체 반환, 아니면 None.

    호출 측에서 None 받으면 401 던지면 됨.
    검증 성공 시 last_used_at 자동 갱신.
    """
    if not raw or len(raw) < _PREFIX_LEN:
        return None
    if not raw.startswith("sv_"):
        return None

    prefix = raw[:_PREFIX_LEN]

    # prefix로 candidates 좁힘 (인덱스 사용)
    stmt = select(ApiKey).where(ApiKey.prefix == prefix)
    result = await db.execute(stmt)
    candidates = result.scalars().all()

    for key in candidates:
        if key.revoked_at is not None:
            continue
        try:
            if _pwd_context.verify(_pepper(raw), key.hash):
                # 사용 시각 업데이트 (별도 트랜잭션 없이 같은 세션)
                key.last_used_at = datetime.now(timezone.utc)
                await db.flush()
                return key
        except Exception:
            continue

    return None
