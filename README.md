# StructVerify Lab v2.0

**도메인 독립형 LLM 기반 사실검증 플랫폼**

Graph + JSON 하이브리드 저장 · 2-Agent 아키텍처 · 도메인 적응형 학습 루프

> "뉴스는 실험 도메인, 시스템은 자기 적응형 범용 검증 플랫폼"

---

## 프로젝트 개요

비정형 텍스트(뉴스, 보고서, 문서)에서 **수치 기반 주장**을 자동으로 추출하고,
KOSIS Open API 등 **공식 통계 데이터**와 비교하여 사실 여부를 검증하는 LLM 기반 플랫폼입니다.

### 핵심 검증 흐름

```
주장 탐지 → 동적 스키마 유도 → 그래프 구축 → 공식 데이터 조회 → 비교 검증 → 설명 생성
```

### 제공 형태

| 형태 | 설명 | 사용 대상 |
|------|------|----------|
| Python 라이브러리 | `verify_text("...")` 직접 호출 | 개발자/연구자 |
| REST API | `/verify/text`, `/verify/document` | 외부 시스템 |
| SaaS 플랫폼 | 웹 UI + 대시보드 | 일반 사용자/기업 |

---

## 빠른 시작

```python
from structverify import verify_text

report = await verify_text("국내 과수 농가의 65세 이상 고령화 비율은 64.2%에 이른다")
print(report.results[0].verdict)  # "match" | "mismatch" | "unverifiable"
```

### API 서버 실행

```bash
pip install -e ".[dev]"
uvicorn api.main:app --reload
# POST http://localhost:8000/verify/text
```

---

## 프로젝트 구조

```
structverify/
├── config/
│   └── default.yaml              # 전체 설정 (LLM, DB, Graph, Snowflake, ...)
│
├── structverify/
│   ├── core/                     # 핵심 오케스트레이션
│   │   ├── schemas.py            # 전체 데이터 모델 (Pydantic)
│   │   ├── pipeline.py           # 13단계 파이프라인 오케스트레이터
│   │   └── config_loader.py      # YAML 설정 로더
│   │
│   ├── preprocessing/            # [전처리 담당] Step 1~2
│   │   ├── extractor.py          # URL/PDF/DOCX → 텍스트 추출
│   │   ├── segmenter.py          # 한국어 문장 분리 + 수치 탐지
│   │   └── sir_builder.py        # SIR Tree 빌더 (Graph Anchor 포함)
│   │
│   ├── detection/                # [검증 담당] Step 3~5
│   │   ├── domain_classifier.py  # Step 3: 도메인 자동 분류
│   │   ├── claim_detector.py     # Step 4: 검증 가능 주장 탐지
│   │   └── schema_inductor.py    # Step 5: Dynamic Schema Induction
│   │
│   ├── graph/                    # [Graph Knowledge Layer] Step 6
│   │   ├── graph_builder.py      # Claim/Evidence Graph 조립
│   │   ├── graph_store.py        # Neo4j/Memgraph 인터페이스
│   │   └── provenance.py         # Provenance(출처) 추적
│   │
│   ├── retrieval/                # [검색 계층] Step 7
│   │   ├── base_connector.py     # 커넥터 추상 인터페이스
│   │   ├── kosis_connector.py    # KOSIS Open API 커넥터
│   │   ├── query_builder.py      # Schema → 검색 쿼리 변환
│   │   └── evidence_subgraph.py  # Evidence 서브그래프 조립
│   │
│   ├── verification/             # [검증 엔진] Step 8
│   │   └── verifier.py           # Deterministic 수치 비교 엔진
│   │
│   ├── explanation/              # [설명 생성] Step 9
│   │   └── explainer.py          # LLM 설명 + Provenance 렌더링
│   │
│   ├── agent/                    # [2-Agent 아키텍처]
│   │   ├── runtime_agent.py      # Agent A: 실시간 검증 (Step 3~9)
│   │   └── builder_agent.py      # Agent B: 비동기 도메인 적응 (Step 11~12)
│   │
│   ├── adaptation/               # [Model Adaptation Layer] Step 11~12
│   │   ├── feedback_store.py     # Human Review/실패 사례 수집
│   │   ├── sample_builder.py     # 학습 샘플 자동 생성
│   │   └── adapter_trainer.py    # LoRA/PEFT Adapter 학습 트리거
│   │
│   ├── storage/                  # [저장 계층]
│   │   ├── raw_storage.py        # S3/MinIO 원본 보존
│   │   ├── db_manager.py         # PostgreSQL OLTP
│   │   └── dwh_manager.py        # Snowflake/BigQuery/ClickHouse DWH
│   │
│   └── utils/
│       ├── logger.py             # 로깅 + OpenTelemetry 기반
│       └── llm_client.py         # LLM API 통합 클라이언트 (HCX/OpenAI)
│
├── api/
│   └── main.py                   # FastAPI REST API 엔드포인트
│
├── tests/
│   └── test_pipeline.py          # 유닛 테스트
│
├── pyproject.toml                # 의존성 관리
└── README.md                     # 이 파일
```

---

## 5계층 아키텍처

| Layer | 구성 요소 | 기술 |
|-------|----------|------|
| **L1: 운영 (OLTP)** | 메타데이터 DB, 비동기 큐, 오브젝트 스토리지 | PostgreSQL, Redis+Celery, S3/MinIO |
| **L2: 검색 (Retrieval)** | 벡터 검색, 키워드 검색, 리랭커, 그래프 검색 | pgvector/Qdrant, Elasticsearch, HCX Reranker, Neo4j |
| **L3: Graph Knowledge** | Claim/Evidence/Provenance 그래프 | Neo4j / Memgraph |
| **L4: 분석 (DWH)** | Data Warehouse, ETL, 대시보드 | **Snowflake** / BigQuery / ClickHouse, Airflow |
| **L5: Model Adaptation** | Feedback Store, Adapter Trainer, Model Registry | PEFT/LoRA, MLflow |

---

## 13단계 파이프라인

| # | 단계 | 담당 | 파일 |
|---|------|------|------|
| 1 | 텍스트 입력 | 전처리 | `preprocessing/extractor.py` |
| 2 | 전처리 + SIR Tree | 전처리 | `preprocessing/sir_builder.py` |
| 3 | 도메인 판별 | Agent A | `detection/domain_classifier.py` |
| 4 | Claim Detection | Agent A | `detection/claim_detector.py` |
| 5 | Dynamic Schema Induction | Agent A | `detection/schema_inductor.py` |
| 6 | Graph Construction | Agent A | `graph/graph_builder.py` |
| 7 | Retrieval + Evidence Subgraph | Agent A | `retrieval/evidence_subgraph.py` |
| 8 | Deterministic Verification | Engine | `verification/verifier.py` |
| 9 | Explanation + Provenance | Agent A | `explanation/explainer.py` |
| 10 | Human Review | 공통 | TODO |
| 11 | Feedback Logging | Agent B | `adaptation/feedback_store.py` |
| 12 | Domain Adaptation Trigger | Agent B | `adaptation/adapter_trainer.py` |
| 13 | Report / API Output | 공통 | `core/pipeline.py` |

---

## 2-Agent 아키텍처

### Agent A: Runtime Verification Agent (`agent/runtime_agent.py`)
실시간 검증 요청을 처리. ReAct 패턴(Thought→Action→Observation)으로 Step 3~9를 오케스트레이션.

### Agent B: Domain Adaptation Agent (`agent/builder_agent.py`)
비동기 학습 담당. 피드백 수집 → 학습 샘플 생성 → Adapter 재학습 → 평가 → 배포.

---

## 참고 논문

| # | 논문 | 저자/학회 | 적용 영역 | 참고 코드 |
|---|------|----------|----------|----------|
| [1] | Docling | IBM Research, 2024 | 전처리 — SIR Tree 설계 | [github.com/DS4SD/docling](https://github.com/DS4SD/docling) |
| [2] | Trafilatura | Barbaresi, ACL 2021 | 전처리 — URL 본문 추출 | [github.com/adbar/trafilatura](https://github.com/adbar/trafilatura) |
| [3] | ClaimBuster | Hassan et al., VLDB 2017 | 검증 — Claim Detection | [github.com/idirlab/claimBuster](https://github.com/idirlab/claimBuster) |
| [4] | FEVER | Thorne et al., NAACL 2018 | 검증 — 3단계 판정 체계 | — |
| [5] | ProgramFC | Pan et al., NAACL 2023 | 검증 — 복잡 주장 분해 | [github.com/mbzuai-nlp/ProgramFC](https://github.com/mbzuai-nlp/ProgramFC) |
| [6] | ReAct | Yao et al., ICLR 2023 | Agent 오케스트레이션 | [github.com/ysymyth/ReAct](https://github.com/ysymyth/ReAct) |
| [7] | FActScore | Min et al., EMNLP 2023 | 검증 — Granularity | [github.com/shmsw25/FActScore](https://github.com/shmsw25/FActScore) |
| [8] | RAG | Lewis et al., NeurIPS 2020 | 검색 계층 패러다임 | [huggingface/transformers](https://github.com/huggingface/transformers) |
| [9] | GraphRAG | arXiv 2501.00309 | Graph 기반 Evidence 검색 | — |
| [10] | AutoSchemaKG | arXiv 2505.23628 | Schema Induction | [github.com/NousResearch/AutoSchemaKG](https://github.com/NousResearch/AutoSchemaKG) |
| [11] | KnowLA | NAACL 2024 | 도메인 적응형 학습 | — |
| [12] | Feedback Adaptation | arXiv 2604.06647 | 비동기 Feedback 루프 | — |
| [13] | Fact Verification on KG | EMNLP Findings 2025 | Graph 기반 사실검증 | — |

---

## 담당별 개발 가이드

### 전처리 담당
- `preprocessing/extractor.py` — URL/PDF/DOCX 텍스트 추출 (TODO 참고)
- `preprocessing/segmenter.py` — 한국어 문장 분리 + 수치 탐지
- `preprocessing/sir_builder.py` — SIR Tree 빌더 (entity_refs, graph_anchor_ids)

### 데이터 적재 담당
- `storage/raw_storage.py` — S3/MinIO 원본 보존
- `storage/db_manager.py` — PostgreSQL OLTP (SIR Tree, Claims, Results)
- `storage/dwh_manager.py` — Snowflake/BigQuery DWH 적재
- `graph/graph_store.py` — Neo4j/Memgraph 그래프 저장

### 데이터 검증 담당
- `detection/claim_detector.py` — Step 4: 주장 탐지 (ClaimBuster 방식 LLM 대체)
- `detection/schema_inductor.py` — Step 5: Dynamic Schema Induction (AutoSchemaKG)
- `graph/graph_builder.py` — Step 6: Claim/Evidence Graph 조립
- `retrieval/kosis_connector.py` — Step 7: KOSIS API 조회
- `verification/verifier.py` — Step 8: Deterministic 수치 비교
- `explanation/explainer.py` — Step 9: LLM 설명 + Provenance

---

## 기술 스택

| 범주 | 기술 |
|------|------|
| Backend | FastAPI (Python) |
| DB (OLTP) | PostgreSQL |
| Graph DB | Neo4j / Memgraph |
| Vector DB | pgvector / Qdrant |
| Cache / Queue | Redis + Celery |
| Object Storage | S3 / MinIO |
| DWH | **Snowflake** / BigQuery / ClickHouse |
| LLM | HCX (NCP) / OpenAI |
| Embedding | HCX Embedding v1/v2 |
| Reranker | HCX Reranker |
| Model Registry | MLflow |
| PEFT | Hugging Face PEFT (LoRA) |
| Observability | OpenTelemetry + Langfuse |

---

## 환경 변수

```bash
export LLM_API_KEY="your-openai-or-hcx-key"
export KOSIS_API_KEY="your-kosis-api-key"
export NEO4J_PASSWORD="your-neo4j-password"
export SNOWFLAKE_ACCOUNT="your-snowflake-account"
export SNOWFLAKE_USER="your-snowflake-user"
export SNOWFLAKE_PASSWORD="your-snowflake-password"
export MINIO_ACCESS_KEY="your-minio-key"
export MINIO_SECRET_KEY="your-minio-secret"
```

---

## 라이선스

내부 프로젝트 — 클라비(Clabi) 협력 과제
