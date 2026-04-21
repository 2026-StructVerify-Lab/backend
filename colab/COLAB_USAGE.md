# StructVerify Colab 실행 가이드

## 권장 Python 버전
- **Python 3.10.x**
- 프로젝트 기준: `>=3.10`
- 저장 파일: `.python-version = 3.10.13`

## 1. Colab에서 브랜치 클론
```bash
REPO_URL="https://github.com/<org>/<repo>.git"
BRANCH_NAME="main"
git clone --depth 1 --branch "$BRANCH_NAME" "$REPO_URL"
```

## 2. 설치
```bash
cd 2026-NLP-04/backend
pip install -r requirements-dev.txt
pip install -e .
```

## 3. 세션 환경변수
```bash
source scripts/env_session.sh
export SV_REPO_URL="$REPO_URL"
export SV_BRANCH="$BRANCH_NAME"
export SV_VERSION="v001"
export SV_SAVE_RESULTS=1
export OPENAI_API_KEY="..."
export KOSIS_API_KEY="..."
sv_print_env
```

## 4. 전체 파이프라인 실행
```bash
bash scripts/run_pipeline.sh \
  --text "국내 과수 농가의 65세 이상 고령화 비율은 64.2%에 이른다" \
  --output-name pipeline_smoke
```

## 5. 레이어별 실행
```bash
bash scripts/run_layer.sh extract --text "..." --output-name layer_extract
bash scripts/run_layer.sh sir --text "..." --output-name layer_sir
bash scripts/run_layer.sh claims --text "..." --output-name layer_claims
bash scripts/run_layer.sh schema --text "..." --output-name layer_schema
bash scripts/run_layer.sh graph --text "..." --output-name layer_graph
bash scripts/run_layer.sh retrieval --text "..." --output-name layer_retrieval
bash scripts/run_layer.sh verify --claim-text "실업률 10%" --claimed-value 10 --official-value 7.2 --output-name layer_verify
bash scripts/run_layer.sh explain --claim-text "실업률 10%" --claimed-value 10 --official-value 7.2 --output-name layer_explain
```

## 6. 결과 저장 위치
기본 저장 위치:
```bash
./run_outputs/<브랜치명>_<버전>/
```

예시:
```bash
./run_outputs/feature-claim-v2_v003/pipeline_smoke.json
./run_outputs/feature-claim-v2_v003/layer_schema.json
```

각 JSON 내부에는 branch, version, saved_at, metadata, payload가 함께 저장된다.

## 7. 별도 결과 레포로 push
```bash
export SV_RESULTS_REPO_URL="https://github.com/<org>/<results-repo>.git"
export SV_RESULTS_REPO_BRANCH="main"
bash scripts/push_results.sh
```

## 8. 저장 없이 실행
```bash
bash scripts/run_pipeline.sh --text "..." --no-save
bash scripts/run_layer.sh claims --text "..." --no-save
```
