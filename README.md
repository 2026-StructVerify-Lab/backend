# StructVerify Backend (v3)

`backend/` 디렉토리는 두 개의 레이어로 나뉜다.

| 디렉토리 | 역할 |
|---|---|
| `structverify/` | **검증 라이브러리** — Pipeline, Agents, Tools, Storage 등 핵심 로직 |
| `sv_platform/` | **FastAPI 플랫폼** — REST API, 인증, Job 관리, DB ORM. `structverify/`를 wrap |

본 문서는 라이브러리 (`structverify/`)에 집중. 플랫폼은 [sv_platform/README.md](sv_platform/README.md) 참조.

---

## 환경

- Python 3.13+
- PostgreSQL 16 + pgvector
- Redis 7
- (옵션) Neo4j 5, Snowflake, Elasticsearch

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

API 키 설정 (env 또는 `.env`):
```
NCP_API_KEY=sk-...            # HCX (NCP CLOVA Studio)
KOSIS_API_KEY=...             # KOSIS Open API
PGVECTOR_DSN=postgresql://structverify:svpass123@localhost:5432/structverify
```

---

## 디렉토리 구조

```
backend/structverify/
├── core/
│   ├── pipeline.py            13단계 통합 (입력 → SIR → 검증 → 설명)
│   ├── schemas.py             전체 데이터 모델 (Pydantic)
│   └── config_loader.py       YAML 설정 로더
│
├── preprocessing/             Step 1~2
│   ├── extractor.py           URL(trafilatura+LLM scraper) / PDF / DOCX / TEXT 추출
│   ├── segmenter.py           kss 문장 분리 + surface signal
│   └── sir_builder.py         SIR Tree 빌더
│
├── detection/                 Step 3~5
│   ├── domain_classifier.py   LLM 도메인 분류 (HCX-DASH-002)
│   ├── candidate_scorer.py    sentence candidate scoring (LLM + heuristic fallback)
│   ├── claim_detector.py      check-worthiness 판별 (HCX-003)
│   └── schema_inductor.py     Dynamic Schema Induction (HCX-007 Structured Outputs)
│                              + value=null 폴백 / value_role 자동 추론 / aggregation 필드
│
├── graph/                     Step 4.5, 6
│   ├── document_graph.py      anchor_year + temporal expression 그래프
│   ├── graph_builder.py       Claim/Evidence Graph 조립
│   ├── graph_store.py         Neo4j 인터페이스 (옵션, 기본 비활성)
│   └── provenance.py          출처 경로 렌더
│
├── retrieval/                 Step 7 (catalog + fetch는 agent tools에서 직접 사용)
│   ├── base.py                BaseDataSource 인터페이스
│   ├── catalog_search.py      pgvector 카탈로그 의미 검색
│   ├── kosis_source.py        KOSIS DataSource (search_catalog + fetch_evidence)
│   ├── kosis_connector.py     KOSIS Open API HTTP 호출
│   ├── evidence_subgraph.py   Evidence 서브그래프
│   ├── query_builder.py       Schema → KOSIS 파라미터
│   └── registry.py            DataSource 레지스트리
│
├── verification/              Step 8 (deterministic 백업 경로 — Phase D에서는 agent loop이 대체)
│   └── verifier.py            수치 비교 + 불일치 유형
│
├── explanation/               Step 9
│   └── explainer.py           LLM 자연어 설명 생성
│
├── agent/                     ★ Phase D Multi-Agent 시스템
│   ├── runtime_agent.py       claim별 process_one_claim 병렬 실행 + dependency level
│   ├── planner.py             ★ Plan 생성 (claim → 검증 전략) — HCX-007
│   ├── reflect.py             ★ 매 iter 다음 action 결정 — HCX-DASH-002
│   ├── loop.py                ★ ReAct loop 본체 (tool 실행 + verdict 합성)
│   ├── dependency_planner.py  ★ sub-claim 실행 레벨 분리 (base=L1, derived=L2)
│   ├── workspace.py           job별 상태 파일 시스템 (verified_facts, sibling, memory)
│   ├── memory.py              memory.md 조작 헬퍼
│   ├── schemas.py             Plan / ActionType / VerdictType / AgentVerdict
│   ├── builder_agent.py       (추후 개발) 사전학습 + 피드백 학습
│   ├── prompts/               planner / reflect prompt 템플릿
│   └── tools/
│       ├── base.py            ToolBase + ToolContext + register_tool
│       ├── catalog_search.py  catalog 검색 (KOSIS pgvector + job-success prepend)
│       ├── fetch_evidence.py  표 fetch + row 매칭 + 후보 폴백
│       ├── calculate.py       수식 계산 (증가율/차이/집계)
│       ├── finish.py          verdict 결정 + loop 종료
│       ├── read_original.py   원문 재독
│       └── explore_catalog.py 카탈로그 카테고리 탐색
│
├── adaptation/                (추후 개발) Builder 학습 파이프라인
│   ├── kosis_crawler.py
│   ├── synthetic_generator.py
│   ├── sample_builder.py
│   ├── adapter_trainer.py
│   └── feedback_store.py
│
├── storage/
│   ├── raw_storage.py         S3/MinIO 원본 보존
│   ├── db_manager.py          PostgreSQL Claims/Results CRUD
│   └── dwh_manager.py         Snowflake/BigQuery DWH (옵션)
│
└── utils/
    ├── logger.py
    └── llm_client.py          LLMClient (HCX v1/v3/Structured) + 전역 rate limiter
```

---

## 라이브러리 사용

```python
from structverify.core.pipeline import VerificationPipeline

pipeline = VerificationPipeline()    # config/default.yaml 자동 로드

# 텍스트 입력
report = await pipeline.run(
    "올 4월 합계출산율은 0.79명으로 지난해 같은 달보다 0.06명 증가했다.",
    source_type="text",
)

# URL 입력 (trafilatura + LLM scraper 자동 추출)
report = await pipeline.run("https://example.com/article", source_type="url")

# 사전 추출 본문 재사용 (sv_platform이 사용)
report = await pipeline.run(url, source_type="url", source_text=already_extracted)

for r in report.results:
    print(r.verdict, r.confidence, r.evidence.official_value, r.explanation)
    if r.supporting_evidence:
        # derived claim의 prev 시점 등 보조 데이터
        for s in r.supporting_evidence:
            print("  supporting:", s.time_period, s.official_value)
```

---

## Phase D Agent Loop 상세

### 1) Document → claim 추출 (Step 1~5)

`core/pipeline.py`의 `run()`이 호출:
1. `extract_text(source, source_type)` — URL/PDF/DOCX/TEXT → markdown raw_text
2. `build_sir(raw_text, src)` — SIR Tree (blocks + sentences)
3. `runtime_agent.process(sir_doc)` 진입

`runtime_agent.process()`:
- Step 3: `classify_domain` (LLM)
- Step 4: `detect_claims` (candidate scoring → check-worthiness)
- Step 4.5: `build_document_temporal_graph` (anchor_year + temporal expression)
- Step 5: `induce_schemas` (LLM JSON schema → ClaimSchema)
  - 한 claim → N sub-claim 분기 (지역별, base/derived 등)
  - **value=null 폴백** — `source_phrase`에서 숫자 max() 복원
  - **value_role 자동 추론** — base / derived_rate / derived_difference / aggregation
- Step 6: `build_claim_graph` (Claim 그래프 + COMPARE 엣지)

### 2) Sub-claim 실행 레벨 분리 (Dependency Planner)

`agent/dependency_planner.py::build_execution_levels(claims)`:

| Level | 포함 | 이유 |
|---|---|---|
| L1 | `value_role in {base, aggregation, None}` | 단독 fetch로 검증 가능. catalog 캐시 공유. |
| L2 | `value_role in {derived_rate, derived_difference}` | base 결과(sibling cache) 의존. L1 끝난 후 실행. |

→ L1 병렬 처리 후 L2 병렬 처리. L1의 verified_facts/sibling_evidence가 L2로 전파됨.

### 3) Per-claim Agent Loop (Step 7~8)

각 claim마다 `_verify_with_agent()` → `agent_loop()`:

```
claim
 │
 ▼
[1] Planner LLM (HCX-007)
    │ - 입력: claim.schema, 본문, anchor_year, anchor_year, 도메인
    │ - 출력: Plan {
    │     claim_type,           # ABSOLUTE/GROWTH_RATE/DIFFERENCE/COMPARISON/RANKING/AGGREGATION
    │     required_data,        # 필요한 데이터 점 명세
    │     initial_steps,        # 권장 액션 시퀀스
    │     fallback,             # 1차 실패 시 대안
    │     calculation_formula,  # 수식 (derived만)
    │   }
    │ - value_role 후처리: schema.value_role 기반 claim_type 강제 정정
    ▼
[2] Reflect Loop (max_iter=10, mode=reflect)
    매 iter:
      a) workspace.read_memory(claim_id) + sibling_evidence inject (iter 1만)
      b) reflect_fn(plan, memory, last_observation, iter_num) → ReflectDecision
         - LLM (HCX-DASH-002) 호출
         - action ∈ {catalog_search, fetch_evidence, calculate, finish,
                     read_original, explore_catalog}
      c) 중복 action 차단 — 같은 (action, input) 2회 연속 → 헛돌이로 판단
      d) Tool 실행 → Observation 기록
      e) Auto-finish 트리거:
         - calculate 성공 후 fetch ≥2 (sibling base 있으면 ≥1) → 즉시 finish
         - LLM이 finish 미호출하고 다음 fetch 시도하는 헛돌이 차단
      f) FINISH 신호 → loop 종료
    ▼
[3] AgentVerdict 생성
    - LLM verdict vs 객관 비교 → 불일치면 합성 verdict로 자동 정정 (N 패치)
    - verdict='comparison' 같은 claim_type 오입 → unverifiable 강등 (P17)
    ▼
[4] runtime_agent에서 primary/supporting Evidence 분리 (P7)
    - primary: claim.schema.time_period와 매칭되는 fetch (없으면 dps[0])
    - supporting: derived claim의 prev 시점 등 (base는 빈 list)
```

### Plan Sequence 가이드 (claim_type별)

| claim_type | 권장 시퀀스 |
|---|---|
| `absolute` | catalog_search → fetch_evidence → finish |
| `growth_rate` | catalog_search → fetch×2 (current+prev) → calculate → finish |
| `difference` | catalog_search → fetch×2 → calculate → finish |
| `comparison` | catalog_search → fetch×2 (두 시점/대상) → finish |
| `aggregation` | catalog_search → fetch×N (N개 시점) → calculate(mean/sum/...) → finish |

---

## Workspace 시스템

매 검증 job마다 `agent_workspace/job_<id>/` 디렉토리 생성. `id`는 `agent.workspace.scope`에 따라:
- `doc_hash` (default): `md5(raw_text)` — 같은 본문 재검증 시 캐시 공유
- `job_id`: API job_id — 매 요청 cold start (멀티-테넌트 격리에 안전)

### 파일 구조
```
agent_workspace/job_<id>/
├── meta.json
├── source.txt                       원본 raw_text (markdown 포함)
├── memory.md                        job-level 메모리 (LLM 입력)
├── verified_facts.json              (indicator, time, population) 키 KOSIS 캐시
├── successful_stat_ids.json         catalog prepend용 stat_id 목록
├── sibling_evidence.json            sent_id 기반 sibling 공유
├── summary.json
└── claims/<claim_id>/
    ├── claim.json
    ├── plan.json                    Planner 출력
    ├── observations/iter_NNN_*.json 매 iter raw 결과
    ├── verdict.json                 최종 AgentVerdict
    ├── log.jsonl
    └── memory.md
```

### 주요 동작
- **verified_facts**: fetch가 성공해서 verdict가 match/mismatch면 저장. 다음 claim이 같은 (indicator, time, population)으로 fetch 시도하면 KOSIS API 호출 없이 cache hit. unit 불일치는 거부.
- **successful_stat_ids**: `fetch_evidence`가 한 번이라도 성공한 stat_id를 prior_success 목록에 추가. `catalog_search`가 다음 검색에서 해당 stat_id를 결과 맨 앞에 prepend (LLM이 같은 표를 우선 고르도록 유도). 라벨: `[같은 job에서 'X 지표' 검증에 사용된 표]` (P5 — 거짓 라벨 제거 + indicator 역추적).
- **sibling_evidence**: `sent_id` 기반. 같은 문장의 base sub-claim이 KOSIS에서 받은 값을 derived sub-claim이 prev 없이 즉시 calculate에 inject 가능.

---

## Tool 시스템 상세

### catalog_search
```python
input: {query: str, category?: list[str]}
output: {candidates: [{id, name, score, ...}, ...]}
```
- pgvector 의미 검색 (KOSIS 메타 임베딩)
- `prior_success` stat_id를 score=1.5로 prepend (P5)

### fetch_evidence
```python
input: {
  candidate_id: str,
  params: {indicator, time_period, population, unit_hint, match_criteria?},
  _candidate_fallbacks?: list[str],   # LLM 미제공 시 catalog observation에서 자동 주입
}
output: {evidence: {value, unit, time_period, stat_table_id, rows, matched_row}}
```
- claim.schema에서 params 자동 보강 (population을 LLM 값보다 우선)
- `match_criteria` carry-over 가드 (P15) — schema.population과 충돌하면 폐기
- 후보 순회 — value=None 응답이어도 다음 후보로 폴백 (P6)
- 성공 시 sibling_evidence + verified_facts 저장

### calculate
```python
input: {formula: str, current?, prev?, aggregation_inputs?: list[float]}
output: {result: float, formula: str}
```
- 증가율: `(current - prev) / prev * 100`
- 차이: `current - prev`
- 집계: `mean` / `sum` / `max` / `min` / `median`

### finish
```python
input: {verdict, confidence, explanation, data_points?}
output: {verdict, ...}
```
- verdict가 enum 외 값(예: `comparison`이라는 claim_type)이면 `unverifiable`로 강등 (P17)
- evidence 없는데 match/mismatch → unverifiable로 강등 (hallucination 가드)
- workspace에 verdict.json + memory.md final 섹션 저장

---

## 설정 (`config/default.yaml`)

핵심 섹션만 발췌. 자세한 내용은 파일 참조.

### Agent
```yaml
agent:
  enabled: true
  workspace:
    backend: "local"
    local_path: "./agent_workspace"
    scope: "doc_hash"              # "doc_hash" | "job_id"
    external_job_id: null           # sv_platform이 자동 주입
    persist_after_job: true
    cleanup_after_days: 7
  loop:
    mode: "reflect"                 # "reflect" | "deterministic"
    max_iterations: 10
    enable_reflection: true
    single_pass_fallback: true
    early_stop_on_confidence: 0.9
  llm:
    plan_model: "structured"        # HCX-007
    reflect_model: "light"          # HCX-DASH-002
    explain_model: "heavy"          # HCX-003
  budget:
    max_tokens_per_job: 100000
    max_concurrent_claims: 3
```

### LLM (전역 rate limit 포함)
```yaml
llm:
  provider: "hcx"
  models:
    heavy:      "HCX-003"
    light:      "HCX-DASH-002"
    structured: "HCX-007"
  temperature: 0.1
  max_tokens: 4096
  api_key_env: "NCP_API_KEY"
  min_call_interval_ms: 600         # 전역 HCX rate limit (~1.6 req/s)
```

### Candidate Detection
```yaml
candidate_detection:
  enabled: true
  threshold: 0.65
  use_surface_signals: true
  teacher_llm_fallback: true
  concurrency: 2                    # LLM 동시 호출 상한
```

### KOSIS
```yaml
kosis:
  base_url: "https://kosis.kr/openapi"
  api_key_env: "KOSIS_API_KEY"
  pgvector_dsn_env: "PGVECTOR_DSN"
  catalog:
    rebuild: false
    embed_batch_size: 100
    min_rows: 1000
```

---

## 데이터 모델 (`core/schemas.py`)

핵심 타입만 정리. 전체는 `schemas.py` 직접 참조 (~340줄).

### Claim
```python
class ClaimSchema:
    indicator: str | None
    time_period: str | None         # "YYYY" | "YYYY-MM"
    unit: str | None
    population: str | None
    value: float | None             # P4 폴백 후
    parent_path: str | None
    prev_value: float | None        # derived 검증용
    prev_time_period: str | None
    prev_phrase: str | None
    value_role: str | None          # "base" | "derived_rate" | "derived_difference" | "aggregation"
    # [추후] aggregation_window, aggregation_time_range
```

### VerificationResult
```python
class VerificationResult:
    claim_id: UUID
    verdict: VerdictType            # match/mismatch/partial/unverifiable
    confidence: float
    evidence: Evidence | None       # primary
    supporting_evidence: list[Evidence]  # derived claim의 prev 등 (P7)
    explanation: str | None
    computed_value: float | None    # calculate 결과
    formula: str | None
```

---

## 테스트

```bash
cd backend
pytest                              # 전체
pytest structverify/agent/          # agent 모듈만
pytest -k "schema_inductor"         # 특정 키워드
```

---

## 개발 노트

### LLM 호출 빈도가 신경 쓰일 때
- `config.llm.min_call_interval_ms` (default 600 → ≈1.6 req/s)
- `config.candidate_detection.concurrency` (default 2)
- HCX 쿼터 늘렸으면 둘 다 하향 가능 (200ms + 4)
- 429 폭주 시 자동 jitter (0~0.7s) + exponential backoff (1/2/4초)

### Workspace 캐시가 stale일 때
- `scope: "doc_hash"`라 같은 본문이면 직전 검증 결과 재사용
- 완전 cold 검증 필요시:
  - `config.agent.workspace.scope: "job_id"`로 전환
  - 또는 `rm -rf agent_workspace/`

### URL 추출이 실패할 때
- 1차 trafilatura가 본문 200자 미만 반환하면 자동 LLM scraper 폴백
- LLM scraper는 Docker sandbox에서 동적 Python 코드 실행 — `preprocessing.sandbox_backend` 설정 확인

### 신규 DataSource 추가 (KOSIS 외)
1. `retrieval/`에 `BaseDataSource` 상속 클래스 작성
2. `@register_datasource("name")` 데코레이터
3. `config/default.yaml`의 `data_sources.enabled`에 `"name"` 추가

---

## 알려진 한계 / 추후 개발

- **Builder Agent (`agent/builder_agent.py`)** — 현재 placeholder. KOSIS Self-Instruct → LoRA fine-tuning 파이프라인은 Phase D 운영 데이터 축적 후 활성화 예정.
- **Aggregation Claim 완전 지원** — `value_role="aggregation"` 분류는 있지만 `aggregation_window` / `aggregation_time_range` 필드 + CalculateTool aggregation_inputs 처리는 미구현 (작업 일시 중단 상태).
- **PDF/DOCX 업로드** — sv_platform 라우트는 있지만 multipart 처리는 Phase 3 예정.
- **Neo4j 활성화** — 현재 `graph.store.enabled=false` 기본. 멀티홉 검증 강화 시 활성.
