#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d "venv" ]]; then
  source venv/bin/activate
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python -m src.eda
python -m src.train

echo "Done. Model: ${ROOT}/models/recommender.pkl"
