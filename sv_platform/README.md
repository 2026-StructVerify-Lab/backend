# sv_platform — Structverify 플랫폼 백엔드

FastAPI + SQLAlchemy (async) + Alembic + JWT/API key 인증.

`structverify/` 라이브러리 위에 얹는 SaaS/API 서비스 레이어.

## 현재 상태 (Phase 1 완료)

| Step | 내용 | 상태 |
|---|---|---|
| 1.1 | 디렉토리 골격 + Pydantic Settings + FastAPI hello | ✅ |
| 1.2 | DB 모델 4종 + Alembic + async session | ✅ |
| 1.3 | API key + JWT + auth middleware + signup/login | ✅ |
| 1.4 | `/v1/verify` (sync) + `/v1/jobs/{id}` | ✅ |

## 디렉토리 구조

```
backend/
├── structverify/                    ← 라이브러리 (기존)
└── sv_platform/                     ← 플랫폼 (이번 단계)
    ├── __init__.py
    ├── config.py                    Pydantic Settings
    ├── db.py                        Async engine + session
    ├── pipeline_runner.py           라이브러리 호출 wrapper
    │
    ├── api/
    │   ├── main.py                  FastAPI app
    │   ├── routes/
    │   │   ├── health.py            GET /health
    │   │   ├── auth.py              POST /v1/auth/signup, /login
    │   │   ├── api_keys.py          /v1/api-keys CRUD
    │   │   ├── verify.py            POST /v1/verify
    │   │   └── jobs.py              GET /v1/jobs, /v1/jobs/{id}
    │   ├── middleware/
    │   │   └── auth.py              get_auth dependency
    │   └── schemas/                 Pydantic 요청/응답
    │       ├── auth.py
    │       ├── api_key.py
    │       └── verify.py
    │
    ├── models/                      SQLAlchemy ORM
    │   ├── base.py                  Base + IdMixin + TimestampMixin
    │   ├── tenant.py
    │   ├── user.py
    │   ├── api_key.py
    │   └── job.py
    │
    ├── auth/
    │   ├── password.py              argon2 hash
    │   ├── api_key.py               키 생성/검증
    │   └── jwt_handler.py           JWT encode/decode
    │
    ├── alembic.ini
    ├── alembic/
    │   ├── env.py
    │   ├── script.py.mako
    │   └── versions/
    │       └── 0001_initial.py      tenants, users, api_keys, jobs
    │
    ├── loaders/                     (Phase 3 — PDF/DOCX/URL)
    ├── datasources/                 (Phase 4 — Custom CSV/DB)
    ├── workers/                     (Phase 2 — ARQ)
    │
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

## 셋업 + 실행

### 1. 의존성 설치

```bash
cd backend
python -m venv .venv && source .venv/bin/activate

# 라이브러리 의존성 (기존)
pip install -e ".[dev]" (처음 이용이면 이거 하세요.)

pip install -r structverify/requirements.txt   # 또는 그쪽 설정대로

# 플랫폼 의존성 (신규)
pip install -r sv_platform/requirements.txt

```

### 2. PostgreSQL 준비

기존 KOSIS catalog DB와 같은 곳에 sv_platform 테이블을 만들어도 무방 (테이블 이름 충돌 없음).

```bash
# Docker로 빠르게
docker run -d --name pg-structverify \
  -e POSTGRES_USER=structverify \
  -e POSTGRES_PASSWORD=structverify \
  -e POSTGRES_DB=structverify \
  -p 5432:5432 pgvector/pgvector:pg16
```

### 3. 환경설정

```bash
cp sv_platform/.env.example sv_platform/.env
# 편집: HCX_API_KEY, KOSIS_API_KEY, AUTH__JWT_SECRET 등
```

### 4. DB Migration

```bash
cd backend
alembic -c sv_platform/alembic.ini upgrade head
```

→ `tenants`, `users`, `api_keys`, `jobs` 테이블 생성됨.

### 5. 서버 실행

```bash
cd backend
uvicorn sv_platform.api.main:app --reload --port 8000
```

→ http://localhost:8000/docs 에서 Swagger UI 자동 생성.

## E2E 테스트

### (1) 회원가입

```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "12345678",
    "tenant_name": "My Company"
  }'
# → {"access_token": "eyJ...", "user": {...}, "tenant": {...}}
```

JWT 받아서 저장:
```bash
TOKEN="eyJ..."
```

### (2) API 키 발급

```bash
curl -X POST http://localhost:8000/v1/api-keys \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My first key"}'
# → {"key": "sv_live_xxx...", "api_key": {...}}
```

raw key 저장:
```bash
SV_KEY="sv_live_xxx..."
```

### (3) 검증 (JWT 또는 API key 둘 다 가능)

```bash
curl -X POST http://localhost:8000/v1/verify \
  -H "Authorization: Bearer $SV_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "text",
    "source_data": "올 10월 합계출산율도 0.76명으로 ...",
    "datasources": ["kosis"]
  }'
# → {"job_id": "uuid", "status": "completed", "poll_url": "/v1/jobs/uuid"}
```

### (4) 결과 조회

```bash
curl http://localhost:8000/v1/jobs/$JOB_ID \
  -H "Authorization: Bearer $SV_KEY"
# → {"id": "...", "status": "completed", "result": {claims: [...]}}
```

## 인증 동작

middleware (`api/middleware/auth.py`)가 두 가지 토큰 모두 처리:

| 헤더 | 분기 | tenant_id 해석 |
|---|---|---|
| `Bearer sv_live_...` | API key | DB lookup (argon2 verify) |
| `Bearer eyJ...` (JWT) | UI 세션 | payload에서 직접 |

`AuthContext` 반환 — 모든 라우트가 이걸 받음:
```python
@router.post("/v1/verify")
async def verify(req, ctx: AuthContext = Depends(get_auth)):
    # ctx.tenant_id 항상 있음
    # ctx.auth_type == "jwt" or "api_key"
```

## Phase 1 한계 (다음 Phase 작업)

- **/v1/verify가 sync** — 요청이 라이브러리 실행 끝까지 기다림. 큰 문서면 timeout 위험.
  Phase 2에서 ARQ로 비동기 전환.
- **PDF/DOCX/URL 미지원** — Phase 3 (Document Loaders).
- **회사 데이터 통합 안 됨** — `datasources=["kosis"]`만 작동.
  Phase 4 (Custom CSV / DB Source).
- **Webhook 발사 안 함** — Phase 2.
- **Rate limit 안 함** — Phase 6.

## Config override

`.env`에서 또는 환경변수로:

```bash
# 단순 필드
APP_ENV=prod DEBUG=false

# 중첩 필드 — `__` 구분자
LLM__PRIMARY_MODEL=HCX-DASH-002
RETRIEVAL__CATALOG_TOP_K=20
AUTH__JWT_SECRET=long-random-string
```

## 라이브러리 통합 방식

`pipeline_runner.py`가 sv_platform과 라이브러리 사이의 유일한 접점.
라이브러리는 sv_platform 존재를 모름:

```python
# sv_platform이 라이브러리 호출하는 방식
from structverify.core.pipeline import VerificationPipeline

pipeline = VerificationPipeline()
report = await pipeline.run(text, "text")
```

설정 주입은 *환경변수*로 (라이브러리가 이미 환경변수 읽는 구조라서 호환):
```python
os.environ["CLOVASTUDIO_API_KEY"] = settings.llm.api_key
os.environ["KOSIS_API_KEY"] = settings.kosis.api_key
```

## Frontend 연결

frontend의 `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

이러면 frontend의 `lib/api.ts`가 mock 대신 실제 backend 호출.
CORS는 `config.py:settings.cors_origins`에서 frontend origin 허용해야 함:
```
CORS_ORIGINS=["http://localhost:3000"]
```
