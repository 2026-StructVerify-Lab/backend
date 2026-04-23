# StructVerify Lab v2.0

**도메인 독립형 LLM 기반 사실검증 플랫폼**

Graph + JSON 하이브리드 저장 · 2-Agent 아키텍처 · 도메인 적응형 학습 루프

> "뉴스는 실험 도메인, 시스템은 자기 적응형 범용 검증 플랫폼"

---

## 프로젝트 개요

비정형 텍스트(뉴스, 보고서, 문서)에서 **수치 기반 주장**을 자동으로 추출하고,
KOSIS Open API 등 **공식 통계 데이터**와 비교하여 사실 여부를 검증하는 LLM 기반 플랫폼입니다.

**핵심 설계 방향: LLM 학습(Fine-tuning) 기반**
- 정규식(Regex) 기반 필터링이나 Rule-based 분류는 **fallback** 보조 신호로만 사용
- 도메인 적응은 KOSIS 메타데이터 기반 Self-Instruct 합성 데이터 → LoRA Fine-tuning으로 수행
- Candidate Detection, Claim Detection, Schema Induction 모두 LLM/학습 모델이 최종 판단

```
[사전학습] KOSIS 메타 수집 → 합성 데이터 생성 → LoRA Fine-tuning
     ↓
주장 탐지 → 동적 스키마 유도 → 그래프 구축 → 공식 데이터 조회 → 비교 검증 → 설명 생성
     ↓
[피드백] Human Review → Feedback Logging → 추가 LoRA 학습
```

| 제공 형태 | 설명 | 사용 대상 |
|----------|------|----------|
| Python 라이브러리 | `verify_text("...")` 직접 호출 | 개발자/연구자 |
| REST API | `/verify/text`, `/verify/document` | 외부 시스템 |
| SaaS 플랫폼 | 웹 UI + 대시보드 | 일반 사용자/기업 |

---

## 빠른 시작

### 1단계: 레포 클론

```bash
git clone https://github.com/your-org/structverify-lab.git
cd structverify-lab
```

### 2단계: 환경변수 설정

```bash
cp infra/.env.example infra/.env
# .env 열어서 KOSIS_API_KEY, LLM_API_KEY는 나중에 개인적으로 받으세욤.
```

- 바꾸실 분들은 알아서 본인 파일 만든 다음에 `.gitignore`에 추가하신 후 이용하세요.

### 3단계: 전체 서비스 올리기

```bash
make dev
```

PostgreSQL(pgvector 포함), Redis, Neo4j, MinIO, Elasticsearch 5개 서비스 업데이트.
추후 AWS에 올리겠습니다. `init_db.sql`이 자동 실행되어 테이블 13개가 즉시 생성됨.

### 4단계: 확인

```bash
make health     # 5개 서비스 헬스체크
make status     # 컨테이너 상태 확인
make psql       # PostgreSQL 직접 접속해서 테이블 확인
```

### 접속 정보

| 서비스 | 주소 | 계정 |
|--------|------|------|
| PostgreSQL | `localhost:5432` | structverify / svpass123 |
| Redis | `localhost:6379` | — |
| Neo4j Browser | http://localhost:7474 | neo4j / svgraph123 |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| Elasticsearch | http://localhost:9200 | — |

### 5단계: Neo4j 인덱스 초기화

```bash
make neo4j-init
```

### DB 테이블 구조

`init_db.sql`에 만들어둔 핵심 테이블은 `documents`, `claims`, `verification_results` 3개가 기본이고, 여기에 `execution_runs`(실행 단위), `artifacts`(중간 산출물), `graph_nodes/edges`(그래프 PG 사본), `kosis_stat_catalog`(메타데이터 캐시), `kosis_data_cache`(응답 캐시), `feedback_events`, `training_jobs`, `model_versions`, `domain_packs` 등 총 13개입니다.

하이브리드 검색을 위한 `kosis_stat_catalog` 테이블에는 pgvector 임베딩 컬럼도 이미 포함되어 있어서, 배치 ETL로 KOSIS 메타데이터를 넣으면 바로 벡터 유사도 검색이 가능합니다.

Snowflake는 로컬에서 못 띄우니까 `init_snowflake.sql`은 나중에 Snowflake 콘솔에서 직접 실행하면 됩니다.

### API 서버 실행

```bash
make install    # 의존성 설치
make api        # FastAPI 서버 시작 (localhost:8000)
```

### 라이브러리 사용

```python
from structverify import verify_text

report = await verify_text("국내 과수 농가의 65세 이상 고령화 비율은 64.2%에 이른다")
print(report.results[0].verdict)  # "match" | "mismatch" | "unverifiable"
```

---

## 개발 순서도

아래 순서대로 개발을 진행합니다. 각 Phase는 이전 Phase 완료 후 시작합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 0: 인프라 & 데이터 적재  ← 박재윤                               │
│                                                                      │
│  docker-compose 구성                                                  │
│    → init_db.sql (테이블 13개)                                        │
│    → kosis_crawler.py (KOSIS 메타 배치 수집)                           │
│    → db_manager.py / dwh_manager.py / graph_store.py 스텁 완성        │
│    → raw_storage.py (MinIO 업로드)                                     │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │ 완료 후
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1A: 전처리 파이프라인  ← 이수민                                  │
│                                                                      │
│  extractor.py (URL/PDF/DOCX 실제 추출)                                │
│    → segmenter.py (kss 문장 분리 + surface signal 부여)                │
│    → sir_builder.py (SIR Tree 변환 + 절대 offset 보정)                 │
│                                                                      │
│  ※ Regex는 has_numeric_surface 같은 fallback 신호만 생성               │
│     실제 candidate 판단은 Phase 1B의 LLM이 담당                        │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │ Phase 0과 병행 가능
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1B: LLM 클라이언트 & Candidate Scorer  ← 김예슬                 │
│                                                                      │
│  llm_client.py (HCX + OpenAI 실제 API 호출 구현)                      │
│    → candidate_scorer.py (Teacher LLM 기반 0~1 점수 계산)              │
│    → claim_detector.py (check-worthiness 2차 판별)                    │
│    → schema_inductor.py (JSON 구조화 스키마 추출)                      │
│    → domain_classifier.py (LLM 도메인 분류)                           │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │ Phase 1A + 1B 완료 후
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 2A: 검증 엔진  ← 신준수                                          │
│                                                                      │
│  kosis_connector.py (KOSIS API 실제 HTTP 호출 + 응답 파싱)              │
│    → query_builder.py (ClaimSchema → KOSIS 파라미터 변환)              │
│    → verifier.py (수치 비교 + 불일치 유형 세분화)                        │
│    → graph_builder.py (Claim/Evidence 노드/엣지 생성)                  │
│    → evidence_subgraph.py (Evidence 서브그래프 조립)                   │
│    → provenance.py (출처 경로 렌더링)                                   │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │ Phase 2A 완료 후
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 2B: Agent 오케스트레이션  ← 김예슬                               │
│                                                                      │
│  runtime_agent.py (ReAct 패턴 Step 3~9 통합)                          │
│    → explainer.py (LLM 자연어 설명 생성)                               │
│    → builder_agent.py (사전학습 + 피드백 학습 루프)                      │
│    → pipeline.py (전체 13단계 통합 + DB/Storage 연동)                  │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │ Phase 2B 완료 후
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 3: LoRA 사전학습 & 피드백 루프  ← 김예슬 + 박재윤                 │
│                                                                      │
│  synthetic_generator.py (Self-Instruct 합성 데이터 생성)               │
│    → sample_builder.py (pretrain / finetune 포맷 변환)                │
│    → adapter_trainer.py (HuggingFace PEFT LoRA 학습/평가/배포)         │
│    → feedback_store.py + DWH 적재 연동                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 전체 파이프라인 상세 (13단계 + 사전학습)

### Step 0: 도메인 사전학습 (Agent B, 서비스 전 1회)
**담당: 김예슬**

```
KOSIS 메타 수집 (kosis_crawler.py)
  → Self-Instruct 합성 데이터 생성 (synthetic_generator.py)
    ↳ claim/schema 샘플: "KOSIS 통계 X는 Y이다" 형태의 positive/negative 쌍
    ↳ candidate detection 샘플: 검증 후보 / 비후보 문장 쌍
  → 학습 포맷 변환 (sample_builder.py)
  → LoRA Fine-tuning (adapter_trainer.py)
  → 평가 → 합격 시 배포 (eval_min_score: 0.85)
```

> 정규식/Rule은 이 단계의 합성 데이터 생성에서만 초기 신호로 사용하고,
> 실제 모델은 LLM이 만든 학습 쌍을 학습하여 스스로 판단합니다.

---

### Step 1: 텍스트 입력 & 추출 — `preprocessing/extractor.py`
**담당: 이수민**

| 입력 형태 | 처리 방식 |
|----------|---------|
| `text` | 그대로 사용 |
| `url` | trafilatura로 본문 추출 |
| `pdf` | PyMuPDF(fitz)로 페이지별 텍스트 추출 |
| `docx` | python-docx로 단락 추출 |

출력: `raw_text: str`

---

### Step 2: 전처리 + SIR Tree 변환 — `preprocessing/sir_builder.py`, `segmenter.py`
**담당: 이수민**

```
raw_text
  → 문단 분할 (re.split \n{2,})
  → 각 문단 → SIRBlock (block_id, type, content, source_offset)
    → 문장 분리 (kss 우선, 정규식 폴백) → Sentence 리스트
      → 각 Sentence에 surface signal 부여:
         - has_numeric_surface: 숫자/단위/비율 포함 여부 (약한 신호, fallback용)
         - candidate_score=0.0, candidate_label=False (초기값, candidate scorer가 채움)
         - graph_anchor_id: 그래프 연결용 노드 ID
      → 절대 char_offset 보정 (block_start + 상대 offset)
    → entity_refs, event_refs 추출 (placeholder NER)
```

출력: `SIRDocument` (doc_id, source_type, blocks[])

> **중요**: segmenter는 `has_numeric_surface` 같은 약한 surface signal만 부여합니다.
> 실제 검증 후보 여부는 Step 4의 candidate_scorer(LLM)가 결정합니다.
> 정규식은 데이터가 없을 때의 fallback 신호에 불과합니다.

---

### Step 3: 도메인 판별 — `detection/domain_classifier.py`
**담당: 김예슬**

```
SIRDocument
  → LLM (HCX-DASH-001 경량 모델) 호출
  → 도메인 분류: "agriculture" | "economy" | "healthcare" | "education" | ...
  → Domain Pack 로드 (domain-packs/{domain}/prompts.yaml)
```

출력: `domain: str`, `sir_doc.detected_domain` 세팅

> Rule-based 키워드 매핑 대신 LLM이 문맥을 보고 직접 분류합니다.

---

### Step 4: Sentence Candidate Detection + Claim Detection — `detection/candidate_scorer.py`, `claim_detector.py`
**담당: 김예슬**

두 단계로 분리된 구조:

```
[4-1] Sentence Candidate Scoring (candidate_scorer.py)
  각 Sentence에 대해:
    → Teacher LLM (HCX-DASH-001) 호출 → 0~1 점수 반환
      - "공식 통계와 연결 가능한 수치 주장인가?"
      - signals: has_quantity, has_time_expr, has_population, has_comparison_expr
    → LLM 실패 시 heuristic fallback (숫자/시점/대상/비교 패턴 가중합)
    → candidate_score, candidate_label, candidate_source 저장

[4-2] Claim Detection / Check-Worthiness (claim_detector.py)
  candidate_score >= threshold (0.65) 문장만 대상으로:
    → LLM (HCX-003 중량 모델) 호출
      - "공식 통계로 검증 가능한 사실 주장인가?"
      - claim_type 분류: increase/decrease/scale/comparison/forecast
    → check_worthy_score >= min_confidence (0.7) 통과 → Claim 객체 생성
```

> **LLM 학습 계획**: candidate_scorer는 현재 teacher LLM 기반이지만,
> 충분한 학습 데이터(Step 0 합성 + 운영 피드백) 누적 후
> 작은 분류 모델(LoRA fine-tuned)로 교체할 예정입니다.

---

### Step 5: Dynamic Schema Induction — `detection/schema_inductor.py`
**담당: 김예슬**

```
Claim (claim_text)
  → LLM (HCX-003) 호출
  → JSON 구조화 추출:
    {
      "indicator": "고령화 비율",
      "time_period": "2023",
      "unit": "%",
      "population": "과수 농가",
      "value": 64.2,
      "comparison_type": "scale"
    }
  → ClaimSchema 객체 생성
  → graph_schema_candidates: 그래프 매핑 후보 노드 타입 제안
```

> [참고] AutoSchemaKG (arXiv 2505.23628) — LLM 기반 동적 스키마 유도

---

### Step 6: Graph Construction — `graph/graph_builder.py`
**담당: 신준수**

```
Claim + ClaimSchema
  → 노드 생성:
    - ClaimNode (claim_id, claim_text)
    - MetricNode (indicator명)
    - TimeNode (time_period)
    - EntityNode (population/대상)
  → 엣지 생성:
    - claim → MEASURED_AT → time
    - claim → BELONGS_TO → metric
    - claim → BELONGS_TO → entity
  → GraphNode[], GraphEdge[] 반환 (Neo4j 저장은 별도)
```

---

### Step 7: Retrieval + Evidence Subgraph — `retrieval/`
**담당: 신준수**

```
ClaimSchema
  → query_builder.py: Schema → KOSIS API 파라미터 변환
    {orgId, tblId, objL1, itmId, prdSe, startPrdDe, endPrdDe}
  → kosis_connector.py: KOSIS Open API 실제 호출
    - /getList: 통계표 목록 조회 (키워드 매칭)
    - /getStatData: 실제 통계 수치 조회
  → evidence_subgraph.py: Evidence 서브그래프 조립
    - EvidenceNode + 출처 엣지 추가
    - ProvenanceRecord 생성
```

출력: `Evidence` (official_value, unit, time_period, raw_response, provenance)

---

### Step 8: Deterministic Verification — `verification/verifier.py`
**담당: 신준수**

```
Claim.schema.value  vs  Evidence.official_value
  → 수치 비교: |claimed - official| / official * 100
  → tolerance 이내 (기본 1.0%): MATCH
  → tolerance 초과: MISMATCH → 불일치 유형 분류
      - VALUE: 단순 수치 오류
      - TIME_PERIOD: 시점 불일치
      - POPULATION: 대상 집단 불일치
      - EXAGGERATION: 과장/축소
  → Evidence 없음: UNVERIFIABLE
```

> **LLM 미개입**: 수치 비교는 hallucination 방지를 위해 deterministic 엔진만 사용합니다.
> 판정 이유 생성(Step 9)만 LLM이 담당합니다.

---

### Step 9: Explanation + Provenance — `explanation/explainer.py`
**담당: 김예슬**

```
Claim + VerificationResult + Evidence
  → LLM (HCX-003) 호출
  → 자연어 설명 생성:
    "KOSIS 통계에 따르면 2023년 과수 농가 고령화 비율은 62.1%로,
     기사의 주장(64.2%)과 약 2.1%p 차이가 있어 불일치로 판정됩니다."
  → provenance.py: 출처 경로 텍스트 렌더링
    "출처: KOSIS 농림어업총조사 (표ID: xxx, 조회일: 2026-04-22)"
```

---

### Step 10: Human Review (TODO)
**담당: 공통**

- 웹 UI에서 검증 결과 수동 확인/수정 인터페이스
- reviewer_verdict 필드에 최종 판정 저장

---

### Step 11~12: 피드백 루프 — `adaptation/`
**담당: 김예슬 + 박재윤**

```
[Step 11] Feedback Logging (builder_agent.py → feedback_store.py)
  Human Review 결과 → FeedbackEvent 저장
  → 누적 건수 >= threshold(10) 시 Step 12 트리거

[Step 12] Domain Adaptation (builder_agent.py → adapter_trainer.py)
  pending 피드백 수집 → 학습 포맷 변환 (sample_builder.py, mode="finetune")
  → LoRA 추가 학습 → 평가 → 배포
```

---

## 2-Agent 아키텍처

### Agent A: Runtime Verification Agent (`agent/runtime_agent.py`)

실시간 검증 요청을 처리하는 메인 Agent입니다. **ReAct (Reason+Act) 패턴**으로 동작합니다.

#### ReAct 패턴이란?

```
Thought → Action (Tool Call) → Observation → Thought → ...
```

LLM이 단순히 답변을 생성하는 것이 아니라, 매 스텝마다 "무엇을 해야 하는지 생각(Thought)"하고,
"구체적인 도구를 호출(Action)"하여 "결과를 관찰(Observation)"한 뒤 다음 행동을 결정합니다.

#### Agent A의 Action 목록 (Step 3~9)

```
┌──────────────────────────────────────────────────────────────────┐
│  Agent A: RuntimeAgent.process(sir_doc)                          │
│                                                                  │
│  Thought: "이 문서의 도메인이 무엇인가?"                             │
│  Action: classify_domain(sir_doc)        → Tool: LLM 분류기       │
│  Observation: domain = "agriculture"                             │
│                                                                  │
│  Thought: "검증할 주장 후보를 찾아야 한다"                            │
│  Action: score_candidate(sentence)       → Tool: Teacher LLM     │
│  Observation: candidate_score=0.82, label=True                   │
│                                                                  │
│  Thought: "후보 문장이 실제 check-worthy한가?"                       │
│  Action: _check_worthiness(sentence)     → Tool: LLM (중량)      │
│  Observation: is_check_worthy=True, score=0.91                   │
│                                                                  │
│  Thought: "주장의 구조화된 스키마를 추출해야 한다"                       │
│  Action: induce_schemas(claims)          → Tool: LLM (중량)      │
│  Observation: {indicator: "고령화비율", value: 64.2, ...}          │
│                                                                  │
│  Thought: "그래프 노드/엣지를 구성해야 한다"                            │
│  Action: build_claim_graph(claims)       → Tool: 내부 로직         │
│  Observation: nodes=[...], edges=[...]                           │
│                                                                  │
│  Thought: "KOSIS에서 공식 수치를 조회해야 한다"                         │
│  Action: build_evidence_subgraph(...)    → Tool: KOSIS API       │
│  Observation: official_value=62.1, unit="%", period="2023"       │
│                                                                  │
│  Thought: "수치를 비교하여 판정해야 한다"                               │
│  Action: verify_claim(claim, evidence)   → Tool: Deterministic   │
│  Observation: verdict=MISMATCH, diff=2.1%                        │
│                                                                  │
│  Thought: "사람이 이해할 수 있는 설명을 생성해야 한다"                    │
│  Action: generate_explanation(claim, result) → Tool: LLM (중량)  │
│  Observation: "KOSIS 기준 62.1%로 불일치 판정..."                   │
└──────────────────────────────────────────────────────────────────┘
```

#### Action → 실제 코드 매핑

| Action 이름 | 호출 함수/클라이언트 | 사용 모델/도구 | 설명 |
|------------|-----------------|------------|-----|
| `classify_domain` | `LLMClient.generate_json()` | HCX-DASH-001 (경량) | 문서 도메인 분류 |
| `score_candidate` | `LLMClient.generate_json()` | HCX-DASH-001 (경량) | 문장별 검증 후보 점수화 |
| `_check_worthiness` | `LLMClient.generate_json()` | HCX-003 (중량) | 실제 검증 가능 주장 판별 |
| `induce_schemas` | `LLMClient.generate_json()` | HCX-003 (중량) | JSON 구조화 스키마 추출 |
| `build_claim_graph` | 내부 로직 | — | 클레임 그래프 구성 |
| `build_evidence_subgraph` | `KOSISConnector` | KOSIS Open API | 공식 통계 조회 |
| `verify_claim` | Deterministic Engine | — | 수치 비교 판정 (LLM 미사용) |
| `generate_explanation` | `LLMClient.generate()` | HCX-003 (중량) | 자연어 설명 생성 |

> **중요**: `verify_claim`(Step 8)은 의도적으로 LLM을 사용하지 않습니다.
> 수치 비교는 LLM이 hallucination을 일으킬 수 있으므로 deterministic 로직으로만 처리합니다.

---

### Agent B: Domain Adaptation Agent (`agent/builder_agent.py`)

두 가지 학습 경로를 관리합니다:

```
[경로 1] 사전학습 (pretrain_domain)
  서비스 시작 전 1회 실행
  KOSIS 메타 → LLM Self-Instruct → 합성 데이터 → LoRA 학습

[경로 2] 피드백 학습 (_trigger_adaptation)
  운영 중 피드백 누적 시 자동 트리거
  Human Review 결과 → 추가 LoRA 학습
```

---

## 정규화/Rule-based 최소화 전략

### 현재 코드에서 Regex/Rule이 남아있는 위치와 역할

| 파일 | Regex/Rule 용도 | 역할 |
|------|---------------|------|
| `segmenter.py` | `NUMERIC_SURFACE_PATTERN` | **fallback 신호만** — candidate scorer가 LLM 판단 못 할 때만 보조 |
| `candidate_scorer.py` | `_score_candidate_heuristic()` | LLM 실패 시 **fallback** — 운영 안정성 확보용 |
| `sir_builder.py` | `_extract_entity_refs()`, `_extract_event_refs()` | 추후 NER 모델로 교체 예정 |

### LLM 학습 기반 흐름

```
1) 초기 bootstrap: Teacher LLM (HCX-003)이 직접 모든 판단 수행
2) 데이터 누적: Teacher 판단 결과를 학습 샘플로 저장
3) Fine-tuning: 누적 샘플로 경량 모델(HCX-DASH) LoRA 학습
4) 교체: 경량 모델이 Teacher를 대체 → 비용 절감 + 속도 향상
```

---

## 인프라 파일 구조

```
infra/
├── docker/
│   └── docker-compose.yml            # 전체 서비스 (PG, Redis, Neo4j, MinIO, ES)
├── scripts/
│   ├── init_db.sql                    # PostgreSQL 테이블 13개 생성 (자동 실행)
│   ├── init_neo4j.cypher              # Neo4j 인덱스/제약조건
│   ├── init_snowflake.sql             # Snowflake DWH 테이블 (수동 실행)
│   └── seed_domain_packs.py           # 기본 Domain Pack 등록
├── Makefile                           # 공통 명령어 (make dev, make test, ...)
├── .env.example                       # 환경변수 템플릿
└── .gitignore
```

---

## 전체 프로젝트 구조

```
structverify-lab/
│
├── backend/                           # FastAPI + 검증 엔진
│   ├── structverify/                  # 핵심 라이브러리 패키지
│   │   ├── core/                      # 오케스트레이션
│   │   │   ├── schemas.py             # 전체 데이터 모델 (Pydantic)
│   │   │   ├── pipeline.py            # 13단계 파이프라인
│   │   │   └── config_loader.py       # YAML 설정 로더
│   │   ├── preprocessing/             # [전처리 담당: 이수민] Step 1~2
│   │   │   ├── extractor.py           # URL/PDF/DOCX → 텍스트 추출
│   │   │   ├── segmenter.py           # 한국어 문장 분리 + surface signal (fallback용)
│   │   │   └── sir_builder.py         # SIR Tree 빌더 (graph anchor 포함)
│   │   ├── detection/                 # [LLM 담당: 김예슬] Step 3~5
│   │   │   ├── domain_classifier.py   # LLM 도메인 분류
│   │   │   ├── claim_detector.py      # candidate filtering + LLM check-worthiness
│   │   │   ├── candidate_scorer.py    # Teacher LLM 기반 sentence candidate scoring
│   │   │   └── schema_inductor.py     # LLM Dynamic Schema Induction
│   │   ├── graph/                     # [검증 담당: 신준수] Step 6
│   │   │   ├── graph_builder.py       # Claim/Evidence Graph 조립
│   │   │   ├── graph_store.py         # Neo4j 인터페이스
│   │   │   └── provenance.py          # 출처 추적
│   │   ├── retrieval/                 # [검증 담당: 신준수] Step 7
│   │   │   ├── base_connector.py      # 커넥터 인터페이스
│   │   │   ├── kosis_connector.py     # KOSIS Open API (getList + getStatData)
│   │   │   ├── query_builder.py       # Schema → KOSIS 쿼리 변환
│   │   │   └── evidence_subgraph.py   # Evidence 서브그래프
│   │   ├── verification/              # [검증 담당: 신준수] Step 8
│   │   │   └── verifier.py            # Deterministic 수치 비교 (LLM 미사용)
│   │   ├── explanation/               # [LLM 담당: 김예슬] Step 9
│   │   │   └── explainer.py           # LLM 설명 생성 + Provenance 렌더링
│   │   ├── agent/                     # [LLM 담당: 김예슬] 2-Agent
│   │   │   ├── runtime_agent.py       # Agent A: ReAct 기반 실시간 검증
│   │   │   └── builder_agent.py       # Agent B: LoRA 사전학습 + 피드백 학습
│   │   ├── adaptation/                # [LLM 담당: 김예슬] Step 0, 11~12
│   │   │   ├── kosis_crawler.py       # KOSIS 메타데이터 수집 (Step 0-1)
│   │   │   ├── synthetic_generator.py # Self-Instruct 합성 데이터 생성 (Step 0-2)
│   │   │   ├── sample_builder.py      # 학습 포맷 변환 (pretrain/finetune)
│   │   │   ├── feedback_store.py      # 피드백 수집 (Step 11)
│   │   │   └── adapter_trainer.py     # LoRA 학습 트리거 (Step 12)
│   │   ├── storage/                   # [데이터 적재 담당: 박재윤]
│   │   │   ├── raw_storage.py         # S3/MinIO 원본 보존 (boto3)
│   │   │   ├── db_manager.py          # PostgreSQL CRUD (SQLAlchemy async)
│   │   │   └── dwh_manager.py         # Snowflake DWH INSERT
│   │   └── utils/
│   │       ├── logger.py              # 로깅
│   │       └── llm_client.py          # LLM 통합 클라이언트 (HCX/OpenAI)
│   ├── api/
│   │   └── main.py                    # FastAPI 엔드포인트
│   ├── tests/
│   │   └── test_pipeline.py           # 유닛 테스트
│   ├── config/
│   │   └── default.yaml               # 설정 파일
│   └── pyproject.toml
│
├── frontend/                          # Next.js SaaS UI (P2)
├── infra/                             # 인프라 코드
├── ml/                                # ML 학습 파이프라인
├── domain-packs/                      # 도메인별 팩 정의
├── docs/                              # 프로젝트 문서
└── .github/                           # CI/CD
```

---

## 13단계 파이프라인 요약 테이블

| # | 단계 | 담당 | 파일 | LLM 사용 |
|---|------|------|------|---------|
| **0** | **도메인 사전학습** | **김예슬** | `kosis_crawler → synthetic_generator → adapter_trainer` | Self-Instruct |
| 1 | 텍스트 입력 & 추출 | 이수민 | `extractor.py` | ✗ |
| 2 | 전처리 + SIR Tree | 이수민 | `sir_builder.py`, `segmenter.py` | ✗ (surface signal만) |
| 3 | 도메인 판별 | 김예슬 | `domain_classifier.py` | ✓ HCX-DASH (경량) |
| 4 | Candidate Scoring + Claim Detection | 김예슬 | `candidate_scorer.py`, `claim_detector.py` | ✓ HCX-DASH + HCX-003 |
| 5 | Dynamic Schema Induction | 김예슬 | `schema_inductor.py` | ✓ HCX-003 (중량) |
| 6 | Graph Construction | 신준수 | `graph_builder.py` | ✗ |
| 7 | Retrieval + Evidence Subgraph | 신준수 | `evidence_subgraph.py`, `kosis_connector.py` | ✗ (API 호출) |
| 8 | Deterministic Verification | 신준수 | `verifier.py` | **✗ (의도적)** |
| 9 | Explanation + Provenance | 김예슬 | `explainer.py` | ✓ HCX-003 (중량) |
| 10 | Human Review | 공통 | TODO | — |
| 11 | Feedback Logging | 김예슬 | `feedback_store.py` | ✗ |
| 12 | Domain Adaptation Trigger | 김예슬 | `adapter_trainer.py` | LoRA 학습 |
| 13 | Report / API Output | 김예슬 | `pipeline.py` | ✗ |
| 적재 | DB/DWH/Graph 저장 | 박재윤 | `db_manager.py`, `dwh_manager.py`, `graph_store.py` | ✗ |

---

## 역할 분담 & TODO 현황

### 이수민 — 전처리 담당

```
preprocessing/extractor.py        — URL(trafilatura), PDF(PyMuPDF), DOCX(python-docx) 텍스트 추출
preprocessing/segmenter.py        — 한국어 문장 분리(kss) + surface signal 추출 (fallback용)
preprocessing/sir_builder.py      — SIR Tree 빌더 (entity_refs, graph_anchor_ids)
```

주요 TODO:
- `extractor.py`: URL/PDF/DOCX 실제 추출 로직 구현
- `segmenter.py`: kss 설치 및 문장 분리 검증
- `sir_builder.py`: entity_refs → NER 모델 교체 (현재 regex placeholder)

### 박재윤 — 데이터 적재 담당

```
storage/raw_storage.py            — S3/MinIO 원본 보존 (boto3)
storage/db_manager.py             — PostgreSQL CRUD (SQLAlchemy async)
storage/dwh_manager.py            — Snowflake INSERT
graph/graph_store.py              — Neo4j driver + Cypher MERGE
adaptation/kosis_crawler.py       — KOSIS 메타데이터 배치 수집 → kosis_stat_catalog 적재
infra/                            — docker-compose, init_db.sql, Makefile 관리
```

주요 TODO:
- `db_manager.py`: SQLAlchemy async engine 초기화 + INSERT 구현
- `dwh_manager.py`: Snowflake connector.connect() + executemany() 구현
- `raw_storage.py`: boto3 MinIO 업로드 구현
- `graph_store.py`: Neo4j driver + Cypher MERGE 구현

### 신준수 — 검증 담당

```
retrieval/kosis_connector.py      — KOSIS API 실제 호출 (getList, getStatData) + 응답 파싱
retrieval/query_builder.py        — Schema → KOSIS 검색 쿼리 변환
retrieval/base_connector.py       — 커넥터 인터페이스 설계
retrieval/evidence_subgraph.py    — Evidence 서브그래프 조립
verification/verifier.py          — deterministic 수치 비교 + 불일치 유형 세분화
graph/graph_builder.py            — Claim/Evidence 그래프 노드/엣지 생성
graph/provenance.py               — 출처 경로 기록 + 텍스트 렌더링
```

주요 TODO:
- `verifier.py`: TIME_PERIOD / POPULATION / EXAGGERATION 불일치 유형 세분화 로직
- `kosis_connector.py`: KOSIS API 실제 HTTP 호출 + 응답 파싱
- `query_builder.py`: ClaimSchema → KOSIS 파라미터 매핑 로직

### 김예슬 — LLM/Agent 담당

```
utils/llm_client.py               — HCX/OpenAI API 실제 호출 구현
detection/claim_detector.py       — check-worthiness 프롬프트 설계/튜닝
detection/schema_inductor.py      — Schema Induction 프롬프트 설계/튜닝
detection/domain_classifier.py    — 도메인 분류 프롬프트
detection/candidate_scorer.py     — teacher LLM 기반 sentence candidate scoring
explanation/explainer.py          — 설명 생성 프롬프트
agent/runtime_agent.py            — Agent A ReAct 오케스트레이션 (Step 3~9)
agent/builder_agent.py            — Agent B 사전학습 + 피드백 학습
adaptation/synthetic_generator.py — Self-Instruct 합성 데이터 생성
adaptation/sample_builder.py      — 학습 포맷 변환
adaptation/adapter_trainer.py     — LoRA 학습/평가/배포
core/pipeline.py                  — 13단계 파이프라인 전체 흐름
```

주요 TODO:
- `llm_client.py`: HCX API (NCP CLOVA Studio) 실제 호출 구현
- `llm_client.py`: OpenAI AsyncClient 실제 호출 구현
- `llm_client.py`: Langfuse 트레이싱 연동
- `candidate_scorer.py`: 학습된 소형 분류 모델 교체 로직 (Phase 3)
- `adapter_trainer.py`: HuggingFace PEFT LoRA 학습 실제 구현

---

## 도메인 지식 습득 전략

| 방법 | 시점 | 설명 | 파일 |
|------|------|------|------|
| RAG (벡터 검색) | Phase 1 | KOSIS 메타데이터 임베딩 → pgvector 유사도 검색 | `kosis_crawler.py` → `kosis_stat_catalog` |
| 프롬프트 Few-shot | Phase 1 | domain-packs/에 도메인 규칙 + 예시 직접 기입 | `domain-packs/news/prompts.yaml` |
| Self-Instruct 사전학습 | Phase 2 | KOSIS 메타 → LLM이 학습 쌍 자동 생성 → LoRA | `synthetic_generator.py` → `adapter_trainer.py` |
| 피드백 기반 학습 | Phase 3 | 검증자 수정 데이터로 추가 LoRA 학습 | `feedback_store.py` → `sample_builder.py` |
| Weak Supervision Candidate Training | Phase 2 | API 매칭 성공 문장 + retrieval/schema 실패 문장 + teacher LLM 신호를 이용해 sentence candidate detector 학습 | `synthetic_generator.py` → `sample_builder.py` → `candidate_scorer.py` |

---

## 제공 NCP API 사용 위치

| API | 파일 | 용도 |
|-----|------|------|
| HCX-003/005/007 | `utils/llm_client.py` | 주장 탐지, 스키마 유도, 설명 생성 |
| HCX-DASH-001/002 | `utils/llm_client.py` | 도메인 분류, sentence candidate scoring (경량) |
| 임베딩 v1/v2 | `utils/embedding_client.py` (TODO) | pgvector 유사도 검색용 임베딩 |
| 리랭커 | `utils/reranker_client.py` (TODO) | 검색 결과 재순위화 |
| RAG Reasoning | `agent/runtime_agent.py` | 복합 근거 종합 판단 |

---

## 기술 스택

| 범주 | 기술 |
|------|------|
| Backend | FastAPI (Python) |
| DB (OLTP) | PostgreSQL + pgvector |
| Graph DB | Neo4j / Memgraph |
| Cache / Queue | Redis + Celery |
| Object Storage | S3 / MinIO |
| DWH | Snowflake / BigQuery / ClickHouse |
| LLM | HCX (NCP) / OpenAI |
| Embedding | HCX Embedding v1/v2 |
| Reranker | HCX Reranker |
| Model Registry | MLflow |
| PEFT | Hugging Face PEFT (LoRA) |
| Observability | OpenTelemetry + Langfuse |
| Frontend | React / Next.js (P2) |

---

## 참고 논문

| # | 논문 | 적용 영역 |
|---|------|----------|
| [1] | Docling (IBM, 2024) | 전처리 — SIR Tree |
| [2] | Trafilatura (ACL 2021) | 전처리 — URL 추출 |
| [3] | ClaimBuster (VLDB 2017) | 검증 — Claim Detection |
| [4] | FEVER (NAACL 2018) | 검증 — 3단계 판정 |
| [5] | ProgramFC (NAACL 2023) | 검증 — 복잡 주장 분해 |
| [6] | ReAct (ICLR 2023) | Agent 오케스트레이션 |
| [7] | FActScore (EMNLP 2023) | 검증 — Granularity |
| [8] | RAG (NeurIPS 2020) | 검색 계층 패러다임 |
| [9] | GraphRAG (arXiv 2501.00309) | Graph Evidence 검색 |
| [10] | AutoSchemaKG (arXiv 2505.23628) | Schema Induction |
| [11] | KnowLA (NAACL 2024) | 도메인 적응형 학습 |
| [12] | Feedback Adaptation (arXiv 2604.06647) | 비동기 Feedback 루프 |
| [13] | Fact Verification on KG (EMNLP Findings 2025) | Graph 사실검증 |
| [14] | Self-Instruct (ACL 2023) | 합성 학습 데이터 생성 |
| [15] | Textbooks Are All You Need (Microsoft, 2023) | 합성 데이터 품질 |
| [16] | Don't Stop Pretraining (ACL 2020) | Domain-Adaptive Pre-training |
| [17] | AdaptLLM (ICLR 2024) | 도메인 Continual Pre-training |

---

## Makefile 명령어

```bash
make dev          # 전체 서비스 시작
make dev-all      # 서비스 + 모니터링 (Langfuse, Grafana)
make down         # 서비스 중지
make reset        # 서비스 중지 + 데이터 초기화
make status       # 컨테이너 상태
make health       # 헬스체크
make logs         # 실시간 로그
make neo4j-init   # Neo4j 스키마 초기화
make install      # Python 의존성 설치
make api          # FastAPI 서버 시작
make test         # 테스트 실행
make lint         # 코드 린트
make psql         # PostgreSQL 접속
make redis-cli    # Redis 접속
```

---

## 라이선스

내부 프로젝트 — 클라비(Clabi) 협력 과제
