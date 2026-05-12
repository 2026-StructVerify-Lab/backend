"""
sv_platform.config — 플랫폼 전역 설정

설정 우선순위 (위가 높음):
  1. 환경변수            (예: POSTGRES_DSN=postgres://...)
  2. .env 파일           (개발 시 편의)
  3. config/*.yaml      (선택, 향후 확장)
  4. 코드의 default 값

[사용법]

    from sv_platform.config import settings

    print(settings.llm.primary_model)        # "HCX-007"
    print(settings.postgres_dsn)              # 환경변수에서 로드됨
    print(settings.app_env)                   # "dev" / "prod"

[중첩 설정 override]

    LLM__PRIMARY_MODEL=HCX-DASH-002 python -m uvicorn ...
                ↑↑                  
                env_nested_delimiter="__"

    이렇게 하면 settings.llm.primary_model이 환경변수로 덮어쓰임.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── LLM 관련 ────────────────────────────────────────────────────────────
class LLMConfig(BaseModel):
    """HCX 등 LLM 호출 설정. structverify 라이브러리에 주입됨."""
    primary_model: str = "HCX-007"
    fallback_model: str = "HCX-DASH-002"
    embedding_endpoint: str = "v2"
    max_tokens: int = 2000
    timeout_seconds: int = 30
    api_key: str = ""               # HCX_API_KEY 환경변수에서 로드


# ── 검증 파이프라인 관련 ────────────────────────────────────────────────
class RetrievalConfig(BaseModel):
    """catalog 검색 / pgvector 관련 설정."""
    catalog_top_k: int = 10
    similarity_threshold: float = 0.5
    pgvector_dsn: str = ""          # 별도 DB일 수도, 같은 DB일 수도


class PipelineConfig(BaseModel):
    """VerificationPipeline 행동 옵션."""
    use_memory: bool = True
    domain_classify: bool = True
    multi_source_fallback: bool = True   # KOSIS 실패 시 Custom source 시도


# ── 외부 시스템 키 ──────────────────────────────────────────────────────
class KOSISConfig(BaseModel):
    """KOSIS Open API."""
    api_key: str = ""               # KOSIS_API_KEY 환경변수에서 로드


# ── 인증/보안 ──────────────────────────────────────────────────────────
class AuthConfig(BaseModel):
    """JWT + API key 검증."""
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_ttl_minutes: int = 60 * 24    # 1일
    # argon2 hashing에 추가 시크릿. 키 hash 시 매번 섞음 (pepper).
    api_key_pepper: str = "change-me-in-production"


# ── 저장소 ─────────────────────────────────────────────────────────────
class StorageConfig(BaseModel):
    """업로드 파일(PDF/DOCX) 저장 위치."""
    type: str = "local"             # local / s3
    local_path: str = "./uploads"
    s3_bucket: str = ""
    s3_region: str = ""


# ── 메인 ───────────────────────────────────────────────────────────────
class AppConfig(BaseSettings):
    """
    플랫폼 전체 설정 진입점.

    Pydantic Settings가 자동으로:
      - .env 파일을 읽고
      - 환경변수로 override 하며
      - 중첩 필드는 `__` 구분자로 접근 가능 (LLM__PRIMARY_MODEL 같이)
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 기본 ──
    app_name: str = "Structverify Platform"
    app_env: str = "dev"            # dev / staging / prod
    app_version: str = "0.1.0"
    debug: bool = True

    # ── 외부 서비스 ──
    postgres_dsn: str = "postgresql+asyncpg://structverify:structverify@localhost:5432/structverify"
    redis_url: str = "redis://localhost:6379/0"

    # ── 중첩 설정 ──
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    kosis: KOSISConfig = Field(default_factory=KOSISConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # ── CORS (UI 도메인 허용) ──
    cors_origins: list[str] = ["http://localhost:3000"]


# 싱글톤 인스턴스 — 앱 전역에서 `from sv_platform.config import settings`
settings = AppConfig()
