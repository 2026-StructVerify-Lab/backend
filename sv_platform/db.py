"""
sv_platform.db — Async SQLAlchemy 엔진 + 세션 의존성

[사용]
    from sv_platform.db import get_session
    
    @router.get("/...")
    async def handler(db: AsyncSession = Depends(get_session)):
        ...

[엔진 생명주기]
- 앱 시작 시 (api/main.py lifespan) engine 생성
- 요청마다 session 새로 발급, 응답 후 자동 close
- 앱 종료 시 engine.dispose()
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from sv_platform.config import settings


# ── 전역 엔진 (lifespan에서 생성/파괴) ──────────────────────────
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine() -> AsyncEngine:
    """
    엔진 + 세션 팩토리 초기화. 앱 시작 시 1회 호출.
    """
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    _engine = create_async_engine(
        settings.postgres_dsn,
        echo=False,       # SQL 로그 (dev only)
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,         # 죽은 connection 자동 감지
    )
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,     # commit 후에도 객체 attribute 접근 가능
    )
    return _engine


async def dispose_engine() -> None:
    """앱 종료 시 호출."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


# ── FastAPI 의존성 ──────────────────────────────────────────────
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI Depends에 넣어서 사용:

        async def handler(db: AsyncSession = Depends(get_session)):
            user = await db.get(User, user_id)
            ...

    예외 발생 시 자동 rollback. 정상 종료 시 자동 close.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized — call init_engine() first")

    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        # commit은 라우트 핸들러가 명시적으로


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """
    스크립트/워커에서 직접 사용할 때:

        async with session_scope() as db:
            ...
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized")

    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
