"""
sv_platform.api.main — FastAPI 애플리케이션 진입점

[실행]
    uvicorn sv_platform.api.main:app --reload --port 8000

[등록된 라우트]
  GET  /                     앱 정보
  GET  /health               헬스체크
  GET  /docs                 Swagger UI

  POST /v1/auth/signup       회원가입 (tenant + user)
  POST /v1/auth/login        로그인 → JWT

  GET    /v1/api-keys        키 목록 (JWT only)
  POST   /v1/api-keys        키 발급 (JWT only, raw key 1회 반환)
  DELETE /v1/api-keys/{id}   키 취소

  POST /v1/verify            검증 제출 (현재 동기, Phase 2부터 비동기)
  GET  /v1/jobs              내 job 목록
  GET  /v1/jobs/{id}         job 상세
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sv_platform.config import settings
from sv_platform.db import dispose_engine, init_engine
from sv_platform.api.routes import api_keys, auth, datasources, health, jobs, verify


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 자원 초기화/해제."""
    print(f"[startup] {settings.app_name} v{settings.app_version} ({settings.app_env})")
    init_engine()
    print(f"[startup] DB engine initialized: {settings.postgres_dsn[:40]}...")
    yield
    await dispose_engine()
    print("[shutdown] DB engine disposed")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Structverify 플랫폼 API — 한국 뉴스 수치 자동 검증 서비스",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(api_keys.router)
app.include_router(verify.router)
app.include_router(jobs.router)
app.include_router(datasources.router)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sv_platform.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )