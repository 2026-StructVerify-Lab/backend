#!/usr/bin/env bash
set -euo pipefail

# 사용 예시:
#   source scripts/env_session.sh
#   bash scripts/run_pipeline.sh --text "국내 과수 농가의 65세 이상 고령화 비율은 64.2%에 이른다"
#   bash scripts/run_pipeline.sh --input-file samples/input.txt --output-name experiment_01

python tools/run_pipeline.py "$@"
