#!/usr/bin/env bash
# Colab/로컬 공용 세션 환경변수 설정 스크립트
# 사용법:
#   source scripts/env_session.sh
#   export SV_REPO_URL="https://github.com/<org>/<repo>.git"
#   export SV_BRANCH="main"
#   export SV_VERSION="v001"
#   export SV_SAVE_RESULTS=1
#   export OPENAI_API_KEY="..."
#   export KOSIS_API_KEY="..."

export SV_REPO_URL="${SV_REPO_URL:-}"
export SV_BRANCH="${SV_BRANCH:-main}"
export SV_VERSION="${SV_VERSION:-v001}"
export SV_SAVE_RESULTS="${SV_SAVE_RESULTS:-1}"
export SV_RESULTS_DIR="${SV_RESULTS_DIR:-./run_outputs}"
export SV_RESULTS_REPO_URL="${SV_RESULTS_REPO_URL:-}"
export SV_RESULTS_REPO_BRANCH="${SV_RESULTS_REPO_BRANCH:-main}"
export SV_CONFIG_PATH="${SV_CONFIG_PATH:-config/default.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

sv_run_dir() {
  printf "%s/%s_%s" "$SV_RESULTS_DIR" "$SV_BRANCH" "$SV_VERSION"
}

sv_print_env() {
  echo "[StructVerify env]"
  echo "  REPO_URL=$SV_REPO_URL"
  echo "  BRANCH=$SV_BRANCH"
  echo "  VERSION=$SV_VERSION"
  echo "  SAVE_RESULTS=$SV_SAVE_RESULTS"
  echo "  RESULTS_DIR=$(sv_run_dir)"
  echo "  CONFIG_PATH=$SV_CONFIG_PATH"
}

mkdir -p "$(sv_run_dir)"
