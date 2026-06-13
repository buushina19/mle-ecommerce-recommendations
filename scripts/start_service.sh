#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/services"

if [[ -d "${ROOT}/venv" ]]; then
  source "${ROOT}/venv/bin/activate"
fi

export MODEL_PATH="${ROOT}/models/recommender.pkl"
export PYTHONPATH="${ROOT}:${ROOT}/services:${PYTHONPATH:-}"

uvicorn ml_service.app:app --host 0.0.0.0 --port 8000
