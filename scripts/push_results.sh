#!/usr/bin/env bash
set -euo pipefail

# 결과물을 별도 레포로 push 하는 스크립트.
# 필요 환경변수:
#   SV_RESULTS_REPO_URL=https://github.com/<org>/<repo>.git
#   SV_RESULTS_REPO_BRANCH=main
# 선택 환경변수:
#   GIT_USER_NAME, GIT_USER_EMAIL

source "$(dirname "$0")/env_session.sh" >/dev/null 2>&1 || true

if [[ -z "${SV_RESULTS_REPO_URL:-}" ]]; then
  echo "SV_RESULTS_REPO_URL 이 비어 있습니다. export 후 다시 실행하세요."
  exit 1
fi

RUN_DIR="$(sv_run_dir)"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "결과 디렉토리가 없습니다: $RUN_DIR"
  exit 1
fi

TMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

git clone --depth 1 --branch "$SV_RESULTS_REPO_BRANCH" "$SV_RESULTS_REPO_URL" "$TMP_DIR/repo"
mkdir -p "$TMP_DIR/repo/$(basename "$SV_RESULTS_DIR")"
DEST="$TMP_DIR/repo/$(basename "$SV_RESULTS_DIR")/$(basename "$RUN_DIR")"
rm -rf "$DEST"
cp -R "$RUN_DIR" "$DEST"
cd "$TMP_DIR/repo"

git config user.name "${GIT_USER_NAME:-StructVerify Bot}"
git config user.email "${GIT_USER_EMAIL:-structverify@example.com}"
git add .
if git diff --cached --quiet; then
  echo "push 할 변경 사항이 없습니다."
  exit 0
fi
git commit -m "Add results for $(basename "$RUN_DIR")"
git push origin "$SV_RESULTS_REPO_BRANCH"
echo "pushed: $SV_RESULTS_REPO_URL -> $SV_RESULTS_REPO_BRANCH"
