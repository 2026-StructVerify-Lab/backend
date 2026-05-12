"""
sv_platform — Structverify 플랫폼 레이어
=============================================

`structverify/` 라이브러리 *위에* 얹는 SaaS/API-key 형태 서비스 레이어.

구성:
  api/         — FastAPI HTTP 인터페이스
  models/      — SQLAlchemy ORM 모델 (Tenant, User, ApiKey, Job, ...)
  auth/        — API key 발급/검증, JWT 핸들러
  loaders/     — 문서 로더 (text/pdf/docx/url → SIR Tree)
  datasources/ — Evidence 소스 추상화 (KOSIS, Custom CSV, Custom DB)
  workers/     — ARQ 비동기 작업자 (검증 job 실행)
  config.py    — Pydantic Settings 기반 전역 설정

이 패키지는 라이브러리(structverify)를 호출만 하지 수정하지 않습니다.
라이브러리는 자체적으로 독립 사용 가능해야 함.
"""
__version__ = "0.1.0"
