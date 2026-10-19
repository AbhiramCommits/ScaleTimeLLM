#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
if [[ -x ".venv/bin/python" && "$PYTHON" == "python" ]]; then
  PYTHON=".venv/bin/python"
fi
"$PYTHON" -m scaletimellm.cli train-verifier \
  --config configs/smoke.yaml \
  --train-problems 140 \
  --eval-problems 45 \
  --k 4 \
  --epochs 18 \
  --seed 0
