# StructVerify Backend (v3)

`backend/` 디렉토리는 두 개의 레이어로 나뉜다.

| 디렉토리 | 역할 |
|---|---|
| `structverify/` | **검증 라이브러리** — Pipeline, Agent, Tools, Retrieval, Storage 등 핵심 로직 |
| `sv_platform/` | **FastAPI 플랫폼** — REST API, 인증, Job 관리, DB ORM. `structverify/`를 wrap |

본 문서는 라이브러리(`structverify/`)에 집중. 플랫폼은 [sv_platform/README.md](sv_platform/README.md) 참조.

---

## 환경

- Python **3.13** (`.venv`의 pyvenv.cfg는 3.13.2)
- PostgreSQL 16 + pgvector
- Redis 7
- (옵션) Neo4j 5, Snowflake, Elasticsearch
- (옵션) Docker — URL extraction의 LLM scraper sandbox 격리용

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

API 키 / DB 설정 (env 또는 `.env`):
```
NCP_API_KEY=<NCP CLOVA Studio key — HCX-003/007/DASH-002/EMB-V2 공용>
KOSIS_API_KEY=<KOSIS Open API key>
PGVECTOR_DSN=postgresql://structverify:svpass123@localhost:5432/structverify
```

> ⚠️ NCP_API_KEY 하나로 LLM 호출(HCX-003/DASH-002/007) + 임베딩(HCX-EMB-V2) + reranker(HCX-RERANKER)를 모두 처리한다. `config/default.yaml`의 `llm.api_key_env`, `embedding.api_key_env`, `reranker.api_key_env`가 전부 `NCP_API_KEY` 가리킴.

---

## 디렉토리 구조

```
backend/structverify/
├── core/
│   ├── pipeline.py            13단계 통합 (입력 → SIR → claim → schema → agent → 검증 → 설명)
│   ├── schemas.py             전체 데이터 모델 (Pydantic)
│   └── config_loader.py       YAML 설정 로더
│
├── preprocessing/             Step 1~2
│   ├── extractor.py           URL(trafilatura + LLMScraper fallback) / PDF / DOCX / TEXT 추출
│   ├── segmenter.py           kss 문장 분리 + 정규표현식 폴백 + surface signal
│   ├── sandbox_backend.py     LLMScraper 동적 Python 실행 격리 (docker/local)
│   └── sir_builder.py         SIR Tree 빌더 (blocks + sentences)
│
├── detection/                 Step 3~5
│   ├── domain_classifier.py   LLM 도메인 분류 (HCX-DASH-002, label 16종 + general fallback)
│   ├── candidate_scorer.py    sentence candidate scoring (LLM + surface heuristic fallback)
│   ├── claim_detector.py      check-worthiness 판별 (HCX-003) — 순위/예측/주관 표현 자동 필터
│   ├── schema_inductor.py     Dynamic Schema Induction (HCX-007 Structured Outputs)
│   │                          + value=null 폴백 / value_role 자동 추론
│   │                          + aggregation_window / aggregation_time_range 추출
│   └── synthetic_generator.py (Builder/Adaptation용 — KOSIS Self-Instruct 생성)
│
├── graph/                     Step 4.5, 6
│   ├── document_graph.py      anchor_year + temporal expression 그래프
│   ├── graph_builder.py       Claim/Evidence Graph 조립 + COMPARE 엣지
│   ├── graph_store.py         Neo4j 인터페이스 (옵션, 기본 비활성)
│   └── provenance.py          출처 경로 렌더
│
├── retrieval/                 Step 7 (catalog + fetch는 agent tools가 직접 호출)
│   ├── base.py                BaseDataSource 인터페이스
│   ├── base_connector.py      ConnectorQuery dataclass (keyword/indicator/time_period/population/extra_params)
│   ├── catalog_search.py      pgvector 카탈로그 의미 검색 (HCX-EMB-V2 임베딩)
│   ├── kosis_source.py        KOSIS DataSource (search_catalog + fetch_evidence + _select_best_row)
│   ├── kosis_connector.py     KOSIS Open API HTTP 호출 + PRD_SE-aware strategy pruning
│   ├── catalog_ranker.py      ★ LLM batch ranking (후보 N개 → 점수 순위, P26)
│   ├── relevance_judge.py     ★ per-table relevance LLM 판단 (P32, 룰 거부 시 fallback)
│   ├── row_matcher.py         ★ row 매칭 LLM rescue (P33c, _select_best_row 0건일 때)
│   ├── dimension_resolver.py  ★ KOSIS 표 차원(itmId/objL) 동적 결정 (P34, cache 있음)
│   ├── evidence_subgraph.py   Evidence 서브그래프
│   ├── query_builder.py       Schema → KOSIS 파라미터
│   └── registry.py            DataSource 레지스트리
│
├── verification/              Step 8 (deterministic 백업 — Phase D에서는 agent loop이 대체)
│   └── verifier.py            수치 비교 + 불일치 유형
│
├── explanation/               Step 9
│   └── explainer.py           LLM 자연어 설명 생성 (HCX-003)
│
├── agent/                     ★ Phase D Multi-Agent 시스템
│   ├── runtime_agent.py       claim별 process_one_claim 병렬 실행 + Level 분리
│   ├── planner.py             ★ Plan 생성 (claim → 검증 전략) — HCX-007
│   ├── reflect.py             ★ 매 iter 다음 action 결정 — HCX-DASH-002
│   ├── loop.py                ★ ReAct loop 본체 (tool 실행 + verdict 합성 + 중복 가드)
│   ├── dependency_planner.py  ★ sub-claim 실행 레벨 분리 (base=L1, derived=L2)
│   ├── workspace.py           job별 상태 파일 시스템
│   │                          (verified_facts / sibling_evidence / fetched_values /
│   │                           successful_stat_ids / failed_stat_ids / memory)
│   ├── memory.py              memory.md 조작 헬퍼
│   ├── schemas.py             ClaimType / ActionType / VerdictType / Plan / AgentVerdict
│   ├── builder_agent.py       (추후 개발) 사전학습 + 피드백 학습
│   ├── prompts/
│   │   ├── planner_prompts.py Plan LLM 프롬프트 템플릿
│   │   └── reflect_prompts.py Reflect LLM 프롬프트 + verdict 가이드
│   └── tools/
│       ├── base.py            ToolBase + ToolContext + @register_tool 데코레이터
│       ├── catalog_search.py  catalog 검색 + job-success prepend + deep/meta_explore 자동 발동
│       ├── fetch_evidence.py  표 fetch + row 매칭 + 후보 폴백 + LLM ranker 적용
│       ├── calculate.py       수식 계산 (증가율/차이/집계, eval 화이트리스트)
│       ├── finish.py          verdict 결정 + 합성 정정 + loop 종료
│       ├── read_original.py   원문 재독
│       ├── explore_catalog.py 카탈로그 카테고리 분포 탐색 (LLM 어휘 학습용)
│       ├── deep_explore.py    ★ T1/T2 — top 표들 sample row preview → LLM row 단서 reasoning
│       ├── meta_explore.py    ★ T1/T2 (기본 모드) — KOSIS getMeta(ITM/OBJ) → 빠른 식별
│       ├── query_rewriter.py  ★ catalog_search query LLM 변형 (row-level keyword → table-friendly)
│       └── replan.py          ★ Plan 자체 갈아끼우기 (per-claim 최대 2회, [[replan_max=2]])
│
├── adaptation/                ★ Builder 학습 파이프라인 (자동 데이터셋 생성)
│   ├── kosis_crawler.py       KOSIS 카탈로그 크롤
│   ├── synthetic_generator.py Self-Instruct 합성 claim 생성
│   ├── sample_builder.py      train/eval split + 정제
│   ├── adapter_trainer.py     LoRA fine-tuning (추후 활성화)
│   ├── feedback_store.py      사용자 피드백 적재
│   └── update_embeddings.py   catalog 임베딩 재구축 스크립트
│
├── storage/
│   ├── raw_storage.py         S3/MinIO 원본 보존 (옵션)
│   ├── db_manager.py          PostgreSQL Claims/Results CRUD
│   ├── dwh_manager.py         Snowflake/BigQuery DWH (옵션)
│   └── init_db.py             초기 스키마 부트스트랩
│
├── memory/                    DocumentWorkingMemory (이수민 main, 도메인/지표 누적)
│   └── ...
│
└── utils/
    ├── logger.py
    └── llm_client.py          LLMClient (HCX v1/v3/Structured Outputs) + 전역 rate limiter
```

---

## 라이브러리 사용

```python
from structverify.core.pipeline import VerificationPipeline

pipeline = VerificationPipeline()    # config/default.yaml 자동 로드

# 텍스트 입력
report = await pipeline.run(
    "올해 4월 출생아 수는 총 2만 717명으로 지난해 같은 달(1만 9059명)보다 8.7% 늘었다.",
    source_type="text",
)

# URL 입력 (trafilatura 1차 + LLMScraper 폴백 자동 추출)
report = await pipeline.run("https://example.com/article", source_type="url")

# 사전 추출 본문 재사용 (sv_platform이 이 경로로 호출 — P13/P18)
report = await pipeline.run(url, source_type="url", source_text=already_extracted)

for r in report.results:
    print(r.verdict, r.confidence, r.evidence.official_value, r.explanation)
    if r.supporting_evidence:
        # derived claim의 prev 시점 등 보조 데이터 (P7)
        for s in r.supporting_evidence:
            print("  supporting:", s.time_period, s.official_value)
```

---

## Phase D Agent Loop 상세

### 1) Document → claim 추출 (Step 1~5)

`core/pipeline.py`의 `run()` 진입:
1. `extract_text(source, source_type)` — URL/PDF/DOCX/TEXT → markdown raw_text
2. `build_sir(raw_text, src)` — SIR Tree (blocks + sentences)
3. `runtime_agent.process(sir_doc)` 호출

`runtime_agent.process()` 내부:
- Step 3: `classify_domain` (HCX-DASH-002)
- Step 4: `detect_claims` (candidate scoring → check-worthiness)
- Step 4.5: `build_document_temporal_graph` (anchor_year + temporal expression)
- Step 5: `induce_schemas` (HCX-007 Structured Outputs → ClaimSchema)
  - 한 claim → N sub-claim 분기 (지역별, base/derived 등)
  - **value=null 폴백** — `source_phrase`에서 숫자 max() 복원
  - **value_role 자동 추론** — base / derived_rate / derived_difference / aggregation
  - **aggregation 필드 추출** — `aggregation_window` (예: "최근 3년" → 3), `aggregation_time_range` (예: ["2022","2023","2024"])
- Step 6: `build_claim_graph` (Claim 그래프 + COMPARE 엣지)

### 2) Sub-claim 실행 레벨 분리 (Dependency Planner)

`agent/dependency_planner.py::build_execution_levels(claims)`:

| Level | 포함 | 이유 |
|---|---|---|
| L1 | `value_role in {base, aggregation, None}` | 단독 fetch로 검증 가능. catalog 캐시 공유. |
| L2 | `value_role in {derived_rate, derived_difference}` | base 결과(sibling cache) 의존. L1 끝난 후 실행. |

→ L1 병렬 처리 후 L2 병렬 처리. L1의 `verified_facts` / `sibling_evidence` / `successful_stat_ids`가 L2로 전파됨.

### 3) Per-claim Agent Loop (Step 7~8)

각 claim마다 `_verify_with_agent()` → `agent/loop.py:agent_loop()`:

```
claim
 │
 ▼
[1] Planner LLM (HCX-007)
    │ - 입력: claim.schema, 본문, anchor_year, 도메인
    │ - 출력: Plan {
    │     claim_type,           # ABSOLUTE/GROWTH_RATE/DIFFERENCE/COMPARISON/RANKING/AGGREGATION
    │     required_data,        # 필요한 데이터 점 명세
    │     initial_steps,        # 권장 액션 시퀀스
    │     fallback,             # 1차 실패 시 대안
    │     calculation_formula,  # 수식 (derived/aggregation)
    │   }
    │ - value_role 후처리: schema.value_role 기반 claim_type 강제 정정
    ▼
[2] Reflect Loop (max_iter=10, mode=reflect)
    매 iter:
      a) workspace.read_memory(claim_id) + sibling_evidence inject (iter 1만)
      b) reflect_fn(plan, memory, last_observation, iter_num) → ReflectDecision
         - LLM (HCX-DASH-002) 호출
         - action ∈ {catalog_search, fetch_evidence, calculate, finish,
                     read_original, explore_catalog, replan}
      c) 중복 action 차단 — 같은 (action, input) 2회 연속 → 헛돌이로 판단,
         3회 연속이면 강제 unverifiable 종료
      d) Tool 실행 → Observation 기록 (claims/<id>/observations/iterNNN_*.json)
      e) Auto-finish 트리거:
         - calculate 성공 후 fetch ≥2 (sibling base 있으면 ≥1) → 즉시 finish
         - LLM이 finish 미호출하고 다음 fetch 시도하는 헛돌이 차단
      f) FINISH 신호 → loop 종료
    ▼
[3] AgentVerdict 생성
    - LLM verdict vs 합성 verdict 일치 검증 → 불일치면 합성으로 자동 정정 (N 패치)
    - verdict='comparison' 같은 claim_type 오입 → unverifiable 강등 (P17)
    - fetch 성공 0건인데 match/mismatch 보고 → unverifiable 강등 (hallucination 가드)
    ▼
[4] runtime_agent에서 primary/supporting Evidence 분리 (P7)
    - primary: claim.schema.time_period와 매칭되는 fetch (없으면 dps[0])
    - supporting: derived claim의 prev 시점 등 (base는 빈 list)
```

### ClaimType별 권장 시퀀스

| claim_type | 시퀀스 |
|---|---|
| `absolute` | catalog_search → fetch_evidence → finish |
| `growth_rate` | catalog_search → fetch×2 (current+prev) → calculate → finish |
| `difference` | catalog_search → fetch×2 → calculate → finish |
| `comparison` | catalog_search → fetch×2 (두 시점/대상) → finish |
| `ranking` | catalog_search → fetch×N (지역별) → 시스템 합성 비교 → finish |
| `aggregation` | catalog_search → fetch×N (N개 시점) → calculate(mean/sum/...) → finish |

---

## Tool 시스템 (Action 11종)

### 검색·탐색

#### `catalog_search`
```python
input: {
  query: str,
  category?: list[str],
  top_k?: int = 15,                    # 기본 15 (cosine 6~15위 정답 포섭)
  force_explore?: bool,                # deep/meta_explore 강제 발동
  explore_mode?: "meta" | "row_preview",  # 기본 "meta" (빠름)
  query_rewrite?: bool,                # LLM이 query 변형 → 합집합 검색
  source?: str,
}
output: {candidates: [{id, name, score, ...}, ...]}
```
- pgvector 의미 검색 + KOSIS 통합검색 API union
- `prior_success` stat_id를 score=1.5로 prepend (P5, 라벨로 indicator 역추적)
- LLM `catalog_ranker` 활성 시 metadata 포함 batch ranking으로 try_ids 재정렬 (P26)
- top1 score 낮거나 top1-top2 gap 작으면 **deep_explore / meta_explore 자동 발동** (T1)

#### `explore_catalog`
- 카탈로그가 실제 어떤 카테고리 어휘를 쓰는지 LLM이 파악 (룰 매핑 없이 self-discover)

#### `deep_explore` (내부 헬퍼 — catalog_search가 자동 호출)
- top N 표의 sample row preview → LLM이 row 단서로 best 표 외삽 추천 (`mode="row_preview"`, 표당 ~22s)

#### `meta_explore` (내부 헬퍼 — catalog_search 기본 모드)
- top N 표의 `getMeta(ITM/OBJ)` 받아 LLM이 정답 표 식별 (표당 ~1s, 권장)

#### `query_rewriter` (내부 헬퍼)
- "체외충격파쇄석기 강원도" 같은 row-level keyword → "시군구별 의료장비"로 변형

### 데이터 조회

#### `fetch_evidence`
```python
input: {
  candidate_id: str,
  params: {indicator, time_period, population, unit_hint, match_criteria?, ...},
  _candidate_fallbacks?: list[str],   # LLM 미제공 시 catalog observation에서 자동 주입
}
output: {evidence: {value, unit, time_period, stat_table_id, rows, matched_row}}
```
- claim.schema에서 params 자동 보강 (population은 LLM 값보다 schema가 우선 — L 패치)
- `match_criteria` carry-over 가드 (P15) — schema.population과 충돌하면 폐기
- 후보 순회 — value=None 응답이어도 다음 후보로 폴백 (P6, _candidate_fallbacks 자동 주입)
- `catalog_ranker` 활성 시 후보 pool을 LLM이 재정렬해 try_ids 결정 (P26)
- `relevance_judge` (P32) — 표 이름과 indicator의 의미 일치 LLM 판단 (룰 거부 시 fallback)
- `dimension_resolver` (P34) — KOSIS getMeta 보고 itmId/objL을 LLM이 동적 결정
- `_select_best_row` — 1차 strict 매칭 실패 시 `row_matcher` LLM rescue (P33c)
- 성공 시 `sibling_evidence` + `verified_facts` + `fetched_values` + `successful_stat_ids` 저장

### 계산

#### `calculate`
```python
input: {
  expression: str,                     # 또는 alias: formula / expr / equation
  variables: {var_name: number, ...},
}
output: {result: float, expression: str, variables: dict}
```
- **eval 화이트리스트**: `+ - * / % **`, `abs`, `round`, `min`, `max`, `sqrt`, `log`, `log10`
- 증가율: `(current - prev) / prev * 100`
- 차이: `current - prev`
- 집계: `mean / sum / max / min / median` (aggregation claim용)

### 종료·재계획

#### `finish`
```python
input: {
  verdict: "match" | "mismatch" | "partial" | "unverifiable",
  confidence: 0.0~1.0,
  explanation: str,
  data_points?: [{indicator, time, resolved_value, source}],
}
output: {verdict, ...}
```
- verdict가 enum 외 값 → `unverifiable`로 강등 (P17, hallucination 가드)
- evidence 없는데 match/mismatch → `unverifiable`로 강등
- workspace에 `verdict.json` + `memory.md` final 섹션 저장

#### `replan`
```python
input: {reason: str}
output: {replan_count, new_plan?}
```
- **호출 조건**: 모든 fetch 후보 실패 + catalog retry/query_rewrite/force_explore 다 시도한 후만
- **호출 효과**: planner LLM이 observation 보고 *완전히 새 plan* 생성 (claim_type 변경 가능)
  - 예: `absolute` → `difference` 변경 (claim이 "증가 수 52"인데 표엔 절대값만)
- **per-claim 최대 2회** (`_REPLAN_MAX_PER_CLAIM=2`)
- replan 후 새 plan 따라 `fetch_evidence`/`calculate` 다시 진행

### `read_original`
- 원문 기사 재독 (`{context_chars: 500}`) — claim 외 정보 필요할 때

---

## Workspace 시스템

매 검증 job마다 `agent_workspace/job_<id>/` 디렉토리 생성. `id`는 `agent.workspace.scope`에 따라:
- `doc_hash` (default): `md5(raw_text)` — 같은 본문 재검증 시 캐시 공유
- `job_id`: API job_id — 매 요청 cold start (멀티-테넌트 격리에 안전)

### 파일 구조
```
agent_workspace/job_<id>/
├── meta.json
├── source.txt                       원본 raw_text (markdown 포함, P18)
├── memory.md                        job-level 메모리 (LLM 입력)
├── verified_facts.json              (indicator, time, population) 키 KOSIS 캐시 (verdict 확정값)
├── fetched_values.json              (stat_id, indicator, time, population) 키 raw fetch 캐시
├── successful_stat_ids.json         catalog prepend + fetch prior 1순위 stat_id 목록
├── sibling_evidence.json            sent_id 기반 base→derived 공유
├── summary.json
└── claims/<claim_id>/
    ├── claim.json
    ├── plan.json                    Planner 출력 (replan 시 덮어씀)
    ├── memory.md
    ├── observations/                매 iter raw 결과
    │   ├── iter001_catalog_search.json
    │   ├── iter002_fetch_DT_xxxx.json
    │   ├── iter003_deep_explore.json
    │   └── ...
    ├── failed_stat_ids.json         per-claim 실패 표 블랙리스트 (P33b, 무한 반복 차단)
    ├── verdict.json                 최종 AgentVerdict
    └── log.jsonl
```

### 캐시 메커니즘

| 캐시 | 키 | 저장 시점 | 사용처 |
|---|---|---|---|
| **verified_facts** | (indicator, time_period, population) | finish의 verdict=match/mismatch | 다음 claim의 fetch lookup 직전. unit 호환 검사 + 파생 접미사 strip (v6.22) |
| **fetched_values** | (stat_id, indicator, time, population) | fetch_evidence 성공 직후 | 같은 claim의 다음 iter 또는 다른 claim이 동일 (stat_id, indicator) 재요청 시 KOSIS 호출 skip (2026-05-26) |
| **successful_stat_ids** | stat_id list | fetch 성공 1회 이상 | catalog_search 결과 맨 앞 prepend + fetch 후보 1순위 |
| **failed_stat_ids** | per-claim stat_id list | fetch가 거부/매칭 실패 | 같은 claim의 다음 catalog_search에서 제외 → 무한 반복 차단 (P33b) |
| **sibling_evidence** | sent_id → {role: evidence} | fetch 성공 직후 | 같은 sent_id의 base가 받은 값을 derived가 prev 없이 calculate에 inject |

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
  max_tokens: 4096                  # 2048→4096 (finish의 explanation 잘림 방지, 2026-05-21)
  api_key_env: "NCP_API_KEY"
  min_call_interval_ms: 600         # 전역 HCX rate limit (~1.6 req/s, 429 대응)

embedding:
  provider: "hcx"
  model: "HCX-EMB-V2"               # 1024-dim
  api_key_env: "NCP_API_KEY"

reranker:
  provider: "hcx"
  model: "HCX-RERANKER"
  api_key_env: "NCP_API_KEY"
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

### KOSIS DataSource
```yaml
data_sources:
  enabled: ["kosis"]
  default_source: "kosis"
  kosis:                            # KOSISDataSource(config=...)에 전달
    catalog_ranker:                 # P26 — LLM batch ranking
      enabled: true
      score_threshold: 0.15
      pool_limit: 20
      max_try: 10
      model_tier: "light"
    relevance_guard:
      enabled: true
      llm_fallback: true            # P32 — 룰 거부 시 LLM 의미 판단 1회
      model_tier: "light"
      row_match_llm_fallback: true  # P33c — _select_best_row 매칭 0건 LLM rescue
      row_match_model_tier: "light"
    dimension_resolver:             # P34 — itmId/objL 동적 결정
      # (config는 default.yaml 참조)
```

### KOSIS (top-level — API URL, catalog 구축용)
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

> ⚠️ `data_sources.kosis`(소스별 ranker/guard 설정)와 top-level `kosis`(API URL + catalog 빌드)는 **별개**.

---

## 데이터 모델 (`core/schemas.py`)

### Claim
```python
class ClaimSchema:
    indicator: str | None
    time_period: str | None         # "YYYY" | "YYYY-MM"
    unit: str | None
    population: str | None
    value: float | None             # P4 폴백 후
    parent_path: str | None
    # derived 지원
    prev_value: float | None
    prev_time_period: str | None
    prev_phrase: str | None
    value_role: str | None          # "base" | "derived_rate" | "derived_difference" | "aggregation"
    # aggregation 지원 (2026-05-21)
    aggregation: str | None         # "mean" | "sum" | "max" | "min" | "median"
    aggregation_window: int | None  # "최근 N년" 의 N
    aggregation_time_range: list[str] | None  # 명시 시점 리스트
```

### ClaimType / ActionType / VerdictType
```python
class ClaimType(str, Enum):
    ABSOLUTE / DIFFERENCE / GROWTH_RATE / COMPARISON / RANKING / AGGREGATION / UNKNOWN

class ActionType(str, Enum):
    CATALOG_SEARCH / EXPLORE_CATALOG / FETCH_EVIDENCE / READ_ORIGINAL /
    CALCULATE / REPLAN / FINISH

class VerdictType(str, Enum):
    MATCH / MISMATCH / PARTIAL / UNVERIFIABLE
```

### VerificationResult
```python
class VerificationResult:
    claim_id: UUID
    verdict: VerdictType
    confidence: float
    evidence: Evidence | None              # primary
    supporting_evidence: list[Evidence]    # derived claim의 prev 등 (P7)
    explanation: str | None
    computed_value: float | None           # calculate 결과
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

## 운영 노트

### LLM 호출 빈도가 신경 쓰일 때
- `config.llm.min_call_interval_ms` (default 600 → ≈1.6 req/s)
- `config.candidate_detection.concurrency` (default 2)
- HCX 쿼터 늘렸으면 둘 다 하향 가능 (200ms + 4 정도)
- 429 폭주 시 자동 jitter (0~0.7s) + exponential backoff (1/2/4초)

### Workspace 캐시가 stale일 때
- `scope: "doc_hash"`라 같은 본문이면 직전 검증 결과 재사용 (정상 동작)
- 완전 cold 검증 필요 시:
  - `config.agent.workspace.scope: "job_id"`로 전환
  - 또는 `rm -rf agent_workspace/`
- 특정 캐시만 무효화하고 싶으면 해당 JSON 파일만 삭제:
  - `agent_workspace/job_<id>/verified_facts.json` (verdict 캐시)
  - `agent_workspace/job_<id>/fetched_values.json` (raw fetch 캐시)

### URL 추출이 실패할 때
- 1차 `trafilatura`가 본문 200자 미만이면 자동 LLMScraper 폴백
- LLMScraper는 Docker sandbox에서 동적 Python 코드 실행 — `preprocessing.sandbox_backend` 설정 확인
- 둘 다 실패하면 빈 문자열 → claim 0건 (pipeline 정상 종료, 결과 없음)

### KOSIS API 429 / 데이터 lag
- KOSIS 응답이 1~2년 지연되는 경우가 많음 (예: 2024-06 기사에서 "최신 데이터"는 2022~2023)
- `_select_best_row`는 명시 시점(prd_target) 정확 매칭만 허용 — 시점 누락 row를 default로 잡지 않음 (패치 3-2). 안전 우선이라 unverifiable로 떨어질 수 있음.

### 신규 DataSource 추가 (KOSIS 외)
1. `retrieval/`에 `BaseDataSource` 상속 클래스 작성
2. `@register_datasource("name")` 데코레이터
3. `config/default.yaml`의 `data_sources.enabled`에 `"name"` 추가
4. `data_sources.<name>` 설정 섹션 추가 (LLM key 등)

---

## 알려진 한계 / 추후 개발

### 미구현
- **Builder Agent (`agent/builder_agent.py`)** — placeholder 상태. KOSIS Self-Instruct → LoRA fine-tuning 파이프라인 자체는 `adaptation/`에 있지만 운영 데이터 축적 후 활성화 예정.
- **PDF/DOCX 업로드** — sv_platform 라우트는 받지만 multipart 처리는 Phase 3 예정 (현재 400 응답).
- **Custom DataSource API** — `custom_csv` / `custom_db` config 자리는 있지만 활성화 미완 (Phase 4).
- **Neo4j 활성화** — 현재 `graph.store.enabled=false` 기본. 멀티홉 검증 강화 시 활성.

### 한계 (설계상)
- **지역명 변경 대응** — "강원도" ↔ "강원특별자치도" (2023 행정구역 개명)는 substring 매칭만 — `'강원도' in '강원특별자치도'`는 False라 매칭 실패 가능. 일부 케이스에서 unverifiable.
- **historical claim (1990년대)** — catalog 임베딩이 표 이름만 기반이라 historical 시리즈 표가 cosine 깊이 묻혀 surface 못 함.
- **외국/지자체 자체 통계** — KOSIS에 없는 데이터는 검증 불가 (의도된 거부).
- **순위/예측만 있는 표현** — claim_detector가 의도적으로 필터 (검증 가능한 수치 없음).
- **LLM hallucination** — planner/reflect가 가짜 candidate_id 박는 케이스. 현재는 fetch 후 next-candidate fallback만 있고 placeholder 자체 필터링은 없음.
- **prev_time fetch 실패** — growth_rate/difference의 prev 시점이 catalog에서 못 찾으면 max_iter 후 unverifiable.

### 시연 추천
- ✅ 최근 1~3년 + 전국/시도 단위 + 단순 absolute / growth_rate
- ✅ "출생아 수", "고용률", "실업률", "소비자물가지수" 등 KOSIS 핵심 통계
- ⚠️ 강원도/전라북도/제주도 시도 (개명 영향), 1990년대 이전, 외국 통계는 피하거나 사전 해명
