"""
sv_platform.api.routes.health — 헬스체크

k8s/load balancer가 호출할 liveness/readiness probe.

[운영 시 추가될 검사 항목]
  - DB 연결 가능?
  - Redis 연결 가능?
  - 외부 API (HCX, KOSIS) 응답?
"""
from fastapi import APIRouter
from pydantic import BaseModel

from sv_platform.config import settings


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """헬스체크 응답."""
    status: str
    env: str
    version: str
    # 향후 확장: db_ok: bool, redis_ok: bool, ...


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    가장 단순한 헬스체크 — 앱이 응답하면 OK.

    향후 DB/Redis/외부 API 상태를 합산해서 종합 status 반환할 예정.
    그때까진 단순히 200 OK + 기본 정보.
    """
    return HealthResponse(
        status="ok",
        env=settings.app_env,
        version=settings.app_version,
    )
