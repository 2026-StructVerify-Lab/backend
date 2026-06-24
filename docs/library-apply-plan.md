# StructVerify 라이브러리 적용 계획

> 작성: 2026-06 · 브랜치: `library/v1/yeseul` · 대상: `backend` (structverify 패키지)
> 전제: 리팩토링(특히 **KOSIS → DataSource 추상화**, PR #57)이 완료되어, 라이브러리화의 토대가 갖춰진 상태.
> 상태 표기: ✅ 완료 · 🚧 부분 · 📋 할 일

---

## 1. 목적

리팩토링으로 엔진이 *도메인 비종속* 구조가 됐으니, `structverify` 를 **독립적으로 설치·사용 가능한 라이브러리** 로 적용한다.
최종 목표는 `pip install structverify` 후 *KOSIS 외 데이터(사내 CSV/DB)로도* config만 바꿔 검증이 되는 것.

---

## 2. 현재 상태 (리팩토링 후 확인됨)

| 항목 | 상태 | 근거 |
|---|---|---|
| DataSource 추상화 | ✅ | `retrieval/base.py::BaseDataSource`, `retrieval/registry.py`(`register_datasource`/`build_datasource`) |
| KOSIS 격리 | ✅ | `kosis_source.py` → `@register_datasource("kosis")` 한 구현으로 분리 |
| 라이브러리 진입점 | 🚧 | `structverify/__init__.py` → `verify_text`, `VerificationPipeline` (문서 입력 `verify_document`는 미노출) |
| 패키징 | 🚧 | `pyproject.toml`(name=structverify, 0.2.0, setuptools) 있음 — 설치/빌드 검증 필요 |
| config 데이터소스 선택 | 🚧 | `data_sources.enabled: ["kosis"]  # 회사: ["custom_db","custom_csv"]` 슬롯만 존재 |
| custom_csv / custom_db | 📋 | registry에 `kosis` 하나뿐 — 구현 없음 |
| LLM provider 추상화 | 🚧 | detection은 `_llm.get_llm_client()` 로 중앙화(부분) |

→ **토대는 됐고, "적용" = 진입점 확정 + 패키징 검증 + custom 소스 구현 + BYO config + 문서.**

---

## 3. 역할 분담

| 담당 | 영역 | 주요 책임 |
|---|---|---|
| **김예슬** (팀장/아키텍트) | 공개 API · LLM · 통합 | 라이브러리 공개 API 확정(`verify_text`/`verify_document`/`VerificationEngine(config)`), LLM provider 추상화, 에이전트·파이프라인 라이브러리화 정합성, 전체 통합·리뷰 |
| **박재윤** (data/infra) | DataSource · 패키징 · 배포 | `custom_csv`·`custom_db` DataSource 구현(`BaseDataSource`+`@register_datasource`), 색인/Onboarding, `pip` 패키징·빌드·의존성, config `data_sources` 스키마 |
| **신준수** (verification/eval) | 검증 · 테스트 | 검증 엔진 진입점 정합, **eval 회귀셋으로 라이브러리화 전후 동작 동일성 검증**, custom 소스 검증 테스트, 스모크/통합 테스트 |

---

## 4. Phase별 작업

각 Phase는 *작은 PR 단위* 로 쪼개고, `refactor:`/`feat:` 커밋 + `[topic]/v1/[이름]` 브랜치 컨벤션을 따른다.

### Phase 1 — 라이브러리 골격 확정  *(담당: 김예슬 + 박재윤)*

- **공개 API 확정** (김예슬)
  - `verify_text(text, config)` 외 **`verify_document(source, source_type, config)`**(url/pdf/docx) 노출.
  - `VerificationEngine(config="...")` 같은 *config 주입형 진입점* 정리. `__init__.py` `__all__` 확정.
- **패키징 검증** (박재윤)
  - `pip install -e .` / 빌드 확인, `pyproject.toml` 의존성·`packages.find` 점검.
  - 스모크: `python -c "import structverify; structverify.verify_text(...)"` 동작.
- **DoD**: 외부 디렉토리에서 `import structverify` → `verify_text`/`verify_document` 실행 성공.
- **안전장치** (신준수): 진입점 호출이 기존 `VerificationPipeline.run()` 과 동일 결과인지 스모크 테스트.

### Phase 2 — custom DataSource 적용 (도메인 독립 입증)  *(담당: 박재윤 + 신준수)*

- **`custom_csv` 구현** (박재윤) — `BaseDataSource` 상속 + `@register_datasource("custom_csv")`. CSV → 카탈로그(임베딩) 색인, `search_catalog`/`fetch_evidence` 구현. column_mapping(value/time/population/indicator)을 config에서 받음.
- **검증 테스트** (신준수) — 작은 CSV 정답셋으로 *KOSIS 아닌 데이터에서 MATCH/MISMATCH가 나오는지* 확인.
- **DoD**: `data_sources.enabled: ["custom_csv"]` 로 바꾸면 사내 CSV 기준 검증 동작.
- (후속) `custom_db`(DSN) + 스키마 introspection 기반 Onboarding은 Phase 4로.

### Phase 3 — BYO config + LLM provider 추상화  *(담당: 김예슬 + 박재윤)*

- **LLM provider 추상화** (김예슬) — `llm.provider`(hcx/openai/local) + `api_key_env` 로 클라이언트 교체. detection의 `_llm` 중앙화를 엔진 전체로 확장.
- **config 스키마 정리** (박재윤) — `llm`/`embedding`/`data_sources`/`domain`/`advanced.prompt_overrides` 를 BYO 제어판 형태로. 비밀값은 env 이름만.
- **DoD**: config 한 파일로 LLM·데이터소스·도메인 지식 주입 가능.

### Phase 4 — 플랫폼 소비 + 문서/예제  *(담당: 김예슬 + 박재윤 + 신준수)*

- **sv_platform 정합** (김예슬) — 현재 in-tree import를 *설치된 라이브러리 소비* 형태로 정리(선택).
- **custom_db(DSN) + Onboarding** (박재윤) — DSN 연결 → 스키마 introspection → 색인 계획.
- **사용 문서/예제** (전원) — 라이브러리 사용법 + "custom DataSource 추가 가이드" + 예제 코드.
- **회귀 최종 확인** (신준수) — eval(oracle/induce)로 라이브러리화 전후 verdict 분포 동일 확인.

---

## 5. 순서 & 안전장치

```
Phase 1 (골격) → Phase 2 (custom_csv) → Phase 3 (BYO config/LLM) → Phase 4 (플랫폼·문서)
```

- **회귀 안전망**: 각 Phase 전후 `eval/`(oracle/induce) 실행 → verdict 분포가 안 바뀌는지 확인(behavior-preserving).
- **작은 PR**: "API 확정" / "패키징" / "custom_csv" 등 단위로 분리, 거대 PR 금지.
- **브랜치/커밋**: `[topic]/v1/[이름]` + `feat:`/`refactor:` 컨벤션. 기존 코드 삭제 대신 버전 주석(팀 규칙).

---

## 6. 결정 필요 사항

| # | 결정 | 비고 |
|---|---|---|
| (a) | 공개 진입점 형태 | `verify_text`/`verify_document` 함수형 ↔ `VerificationEngine(config)` 객체형 (둘 다 제공?) |
| (b) | custom 소스 우선순위 | `custom_csv`(쉬움) 먼저 → `custom_db`(DSN) 후속 — 본 계획 기본값 |
| (c) | sv_platform 소비 방식 | in-tree import 유지 ↔ 설치 패키지로 분리 |
| (d) | LLM provider 범위 | 우선 hcx+openai ↔ local(vLLM)까지 |

---

## 7. 즉시 다음 할 일 (Phase 1 시작)

- [ ] (김예슬) `verify_document` 시그니처 설계 + `__init__.py __all__` 확정
- [ ] (박재윤) `pip install -e .` 빌드·의존성 점검
- [ ] (신준수) 진입점 스모크 테스트 작성 (`verify_text`/`verify_document` == `pipeline.run`)

---

## 8. 라이브러리 사용법 · 설정 (예시 파일)

> 예시 파일: `config/default_example.yaml`(설정), `examples/use_library.py`(코드).
> `config/default.yaml` 은 `.gitignore` 대상(키 들어감)이라, *예시는 `*_example.yaml`* 로 공유한다.

### 설치
```bash
pip install -e .            # (+ pip install openai  — upstage/openai provider 쓸 때)
```

### 필요 환경변수 (.env)
`.env.example` 은 저장소 규칙상 ignore라, 필요한 키를 여기 정리한다.

| env | 용도 |
|---|---|
| `NCP_API_KEY` | HCX (LLM·임베딩·reranker 공용) |
| `UPSTAGE_API_KEY` | Upstage(Solar) LLM |
| `OPENAI_API_KEY` | OpenAI provider |
| `KOSIS_API_KEY` | KOSIS 데이터 소스 |
| `PGVECTOR_DSN` | 카탈로그(pgvector) |
| `CUSTOM_DB_DSN` | custom_db 데이터 소스 |

### LLM provider 전환
`config/default.yaml` 의 `llm.provider` 를 `hcx | openai | upstage` 로 변경. HCX 키가 없으면 Upstage(Solar, OpenAI 호환):

```yaml
llm:
  provider: "upstage"
  api_key_env: "UPSTAGE_API_KEY"
  base_url: "https://api.upstage.ai/v1"
  models: { heavy: "solar-pro", light: "solar-mini", structured: "solar-pro" }
```

※ 임베딩/카탈로그는 아직 HCX-EMB 기준 → Upstage만으로 전체 E2E는 임베딩 전환(재임베딩) 후 가능 (Phase 3, 박재윤).

### 사용 (코드)
```python
import asyncio, structverify
report = asyncio.run(structverify.verify_text("올해 출생아 수는 2만 명이다"))
# 문서 입력:  structverify.verify_document(url, source_type="url")
```

---

## 9. 진행 현황

- ✅ [Phase 1] 공개 API `verify_document` 추가 (`pipeline.py` / `__init__.py`)
- ✅ [Phase 3 일부] LLM provider `upstage`(Solar) 추가 (`llm_client.py`)
- ✅ 예시 파일 — `config/default_example.yaml`(Upstage 반영), `examples/use_library.py`
- 📋 나머지는 4절 Phase·역할분담 참고 (이슈로 트래킹)

> 브랜치 `library/v1/yeseul` 에서 진행 중. 이 문서는 계획 + 진행 현황 기록.
