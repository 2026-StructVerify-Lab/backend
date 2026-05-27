# sv_platform — StructVerify FastAPI 플랫폼 (v3)

`structverify/` 라이브러리 위에 얹는 SaaS/API 서비스 레이어.

FastAPI + SQLAlchemy (async) + Alembic + JWT/API key 인증.

라이브러리 자체 흐름은 [../README.md](../README.md) (backend/README), 검증 라이브러리 상세는 [../../README.md](../../README.md) (루트 README) 참조.

---

## 현재 상태

| Phase | 내용 | 상태 |
|---|---|---|
| 1.1 | 디렉토리 + Pydantic Settings + FastAPI hello | ✅ |
| 1.2 | DB 모델 4종 + Alembic + async session | ✅ |
| 1.3 | API key + JWT + signup/login | ✅ |
| 1.4 | `/v1/verify` (sync) + `/v1/jobs/{id}` | ✅ |
| 2.1 | **BackgroundTask 비동기 실행** (Phase 2) | ✅ |
| 2.2 | **partial workspace 폴링** (실시간 plan/trace 노출) | ✅ |
| 2.3 | **URL 입력 지원 + 사전 추출 + Job.source_data 즉시 저장** (P13, P18) | ✅ |
| 3 | multipart PDF/DOCX 업로드 | 🚧 미구현 |
| 4 | Custom DataSource 등록 (CSV/사내 DB) | 🚧 미구현 |

---

## 디렉토리 구조

```
backend/sv_platform/
├── api/
│   ├── main.py                    FastAPI app + lifespan + CORS
│   ├── routes/
│   │   ├── health.py              GET /health
│   │   ├── auth.py                POST /v1/auth/signup, /login, /v1/auth/me
│   │   ├── api_keys.py            /v1/api-keys CRUD
│   │   ├── verify.py              POST /v1/verify (text/url)
│   │   ├── jobs.py                GET /v1/jobs, /v1/jobs/{id}
│   │   │                          - 진행 중일 때 workspace partial 노출
│   │   │                          - claim 진행률 역추정 → progress 정정
│   │   └── datasources.py         (Phase 4) Custom DataSource CRUD 스텁
│   ├── middleware/
│   │   └── auth.py                get_auth dependency (JWT or API key)
│   └── schemas/                   Pydantic 요청/응답
│       ├── auth.py
│       ├── api_key.py
│       └── verify.py              VerifyRequest, JobOut
│
├── models/                        SQLAlchemy ORM
│   ├── base.py                    Base + IdMixin + TimestampMixin
│   ├── tenant.py
│   ├── user.py
│   ├── api_key.py
│   └── job.py                     Job(id, tenant_id, status, source_type,
│                                       source_data, source_uri, result, progress, ...)
│
├── auth/
│   ├── password.py                argon2 hash
│   ├── api_key.py                 키 생성/검증
│   └── jwt_handler.py             JWT encode/decode
│
├── loaders/
│   └── workspace_reader.py        agent_workspace에서 partial claim/plan/trace 읽기
│                                  - source_text whitespace 정규화 매칭
│                                  - created_after로 과거 잡 claim 제외
│
├── alembic/
│   ├── env.py
│   └── versions/                  마이그레이션
│
├── pipeline_runner.py             ★ structverify 라이브러리 호출 wrapper
│                                  - URL 사전 추출 (P13)
│                                  - external_job_id 주입 (P8 scope=job_id 모드)
│                                  - source_text fallback 매칭 정보 전달
│                                  - 라이브러리 logger handler 부착해 progress 추출
│
├── config.py                      Pydantic Settings
├── db.py                          Async engine + session
├── requirements.txt
└── .env.example
```

---

## API 엔드포인트

### 인증
```
POST /v1/auth/signup          { email, password } → JWT
POST /v1/auth/login           { email, password } → JWT
GET  /v1/auth/me              현재 사용자 정보
```

### API Keys
```
POST   /v1/api-keys           새 API key 발급
GET    /v1/api-keys           내 API key 목록
DELETE /v1/api-keys/{id}      API key 삭제
```

### Verify
```
POST /v1/verify
  Body (JSON):
    {
      "source_type": "text" | "url" | "pdf" | "docx",
      "source_data": "...",            # text 입력
      "url": "https://...",            # url 입력
      "datasources": ["kosis"],
      "callback_url": "https://my-app/webhook"  (선택)
    }
  Response:
    JobOut { id, status="pending", source_type, ... }

  PDF/DOCX는 multipart 라우트 필요 (Phase 3 미구현 → 400 응답)
```

### Jobs
```
GET /v1/jobs?limit=20&offset=0       내 job 리스트
GET /v1/jobs/{id}                    job 상세
  - status=pending/running이면 agent_workspace에서 partial 읽어 result에 enrich
  - claim별 plan/trace/verdict까지 실시간 노출
  - 진행률 역추정 → progress 정정
```

---

## Background 실행 흐름 (run_verification_background)

```python
# verify.py
job = Job(status="pending", source_type=..., source_data=..., source_uri=...)
db.add(job); await db.commit()
background_tasks.add_task(
    run_verification_background,
    job_id=job.id,
    source_type=...,
    source_data=..., source_uri=...,
    datasources=...,
)
return JobOut.model_validate(job)        # 즉시 응답
```

`pipeline_runner.run_verification_background()`의 흐름:

```
1) job.status='running', progress=5, current_step='초기화'
2) structverify logger에 progress handler 부착
   - [Agent A] Step 3 classify_domain → progress=15
   - [Agent A] Step 4 detect_claims → progress=25
   - ...
3) _check_input + _inject_env
4) [URL일 때만] 사전 추출 (P18)
   a) current_step='URL 본문 추출 중', progress=8
   b) await extract_text(source_uri, URL)
   c) Job.source_data = extracted_markdown
      current_step='본문 추출 완료', progress=12
5) config.agent.workspace.external_job_id = str(job_id) 주입 (scope=job_id 모드용)
6) await pipeline.run(source, source_type, source_text=pre_extracted)
7) _build_response(report, job_id, source_text=effective_text)
   - workspace_reader가 source_text로 디렉토리 매칭
   - evidence 경량화 (KOSIS raw response 제외)
   - supporting_evidence 같이 직렬화
8) job.status='completed', result=full_dict, progress=100
   - URL일 때 Job.source_data 최종 보강
```

---

## Workspace 매칭 정책

`workspace_reader._find_job_dir()` 매칭 우선순위:

```
1. agent_workspace/job_<job_id>/  (정확 매칭 — scope=job_id 모드)
2. source_text 일치 매칭 (scope=doc_hash 모드)
   - normalize_ws(source.txt) == normalize_ws(Job.source_data)
   - 같은 텍스트로 여러 워크스페이스가 있으면 claims/ mtime이 최신인 것
3. job_id prefix 매칭 (source_text 없을 때만)
```

**(P18 변경)** workspace의 `source.txt`는 이제 `sir_doc.raw_text`(P10 추가 필드, 원본 markdown 보존)로 저장됨. 이전엔 sir_doc의 sentence를 공백 join한 결과라 URL 추출본의 markdown 구조와 매칭 실패하던 버그 해결.

---

## 실시간 partial 응답

`GET /v1/jobs/{id}` 가 progress 중일 때:

```python
if job.status in ("pending", "running") and not out.result:
    partial = read_partial_job_workspace(
        str(job_id),
        source_text=job.source_data,         # URL일 땐 P13 사전 추출본
        created_after=job.created_at.timestamp(),
    )
    if partial:
        out.result = partial                 # claims + plan + trace (verdict는 있는만큼)
        # progress 역추정 — DB가 5%에 멈춰있어도 claim 상태로 추론
        _enriched_progress, _enriched_step = _estimate_progress_from_partial(partial)
```

### 진행률 역추정 (`_estimate_progress_from_partial`)

| 조건 | progress | step |
|---|---|---|
| claims=[] | 30% | "검증 가능 주장 탐지" |
| plan만 있음 | 50% | "Claim 그래프 빌드" |
| trace 일부 | 50~65% | "공식 통계 조회" |
| verdict 일부 | 65~90% | "수치 검증" |
| 전 verdict 완료 | 95% | "근거 설명 생성" |

---

## DB 모델 (`models/`)

```sql
tenants(id, name, created_at)
users(id, tenant_id, email, password_hash, created_at)
api_keys(id, tenant_id, key_hash, label, last_used_at, created_at)
jobs(
  id UUID,
  tenant_id UUID,
  api_key_id UUID,
  status            VARCHAR(20),         -- pending | running | completed | failed
  source_type       VARCHAR(20),         -- text | url | pdf | docx
  source_data       TEXT,                -- text 본문 또는 URL 추출본 (P13)
  source_uri        TEXT,                -- URL 또는 파일 경로
  datasources       JSON,
  callback_url      TEXT,
  progress          INT,
  current_step      VARCHAR(100),
  result            JSON,                -- 최종 응답 dict
  error             TEXT,
  started_at        TIMESTAMP,
  completed_at      TIMESTAMP,
  created_at        TIMESTAMP
)
```

---

## 셋업 + 실행

### 1. 의존성

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                          # structverify 라이브러리
pip install -r sv_platform/requirements.txt      # FastAPI + DB
```

### 2. 환경변수 (`.env`)

```env
# DB
SV_DATABASE_URL=postgresql+asyncpg://structverify:svpass123@localhost:5432/structverify

# JWT
SV_JWT_SECRET=change-me
SV_JWT_ALG=HS256
SV_JWT_EXP_HOURS=24

# LLM / KOSIS (structverify 라이브러리용)
NCP_API_KEY=sk-...
KOSIS_API_KEY=...
PGVECTOR_DSN=postgresql://structverify:svpass123@localhost:5432/structverify
```

### 3. DB 마이그레이션

```bash
cd backend/sv_platform
alembic upgrade head
```

### 4. 서버 기동

```bash
cd backend
uvicorn sv_platform.api.main:app --reload --port 8000
```

- 문서: http://localhost:8000/docs
- 헬스: http://localhost:8000/health

---

## 인증 흐름

### JWT (웹 UI용)
```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"pw"}'
# → { "token": "eyJ..." }

curl -H "Authorization: Bearer eyJ..." http://localhost:8000/v1/jobs
```

### API Key (외부 시스템용)
```bash
# 1) JWT로 키 발급
curl -X POST http://localhost:8000/v1/api-keys \
  -H "Authorization: Bearer eyJ..." \
  -d '{"label":"prod-key"}'
# → { "key": "sk-..." }   (한 번만 노출, 나머지는 hash만 저장)

# 2) 이후 API key로 호출
curl -H "Authorization: Bearer sk-..." http://localhost:8000/v1/verify ...
```

`get_auth` 미들웨어가 Bearer 토큰을 JWT/API key 둘 다 시도.

---

## 알려진 한계 / 추후

- **PDF/DOCX multipart 업로드** — 라우트는 있지만 400 응답. `loaders/` 패키지에 처리 코드 추가 + `verify.py`에서 분기 필요.
- **ARQ/Celery 분산 큐** — 현재 BackgroundTask는 단일 uvicorn worker 내 asyncio. 운영 부하 시 분산 큐로 교체 필요.
- **Webhook 발송** — `callback_url` 필드는 받지만 실제 POST 발송은 미구현.
- **Streaming** — 현재 폴링 방식. SSE/WebSocket으로 push 전환 고려.
- **멀티-테넌트 quota** — tenant별 LLM 토큰/job 수 제한 미구현.
