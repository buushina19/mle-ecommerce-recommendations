#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p mlruns models

export MLFLOW_TRACKING_URI="sqlite:///${ROOT}/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="${ROOT}/mlruns"

echo "MLflow UI: http://127.0.0.1:5000"
mlflow ui \
  --backend-store-uri "${MLFLOW_TRACKING_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port 5000
