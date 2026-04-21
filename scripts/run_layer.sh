#!/usr/bin/env bash
set -euo pipefail

# 사용 예시:
#   source scripts/env_session.sh
#   bash scripts/run_layer.sh extract --text "..."
#   bash scripts/run_layer.sh claims --input-file samples/input.txt
#   bash scripts/run_layer.sh verify --claim-text "실업률 10%" --claimed-value 10 --official-value 7.2

python tools/run_layer.py "$@"
