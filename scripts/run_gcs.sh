#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

if [ -z "${BUCKET_NAME:-}" ]; then
  echo "BUCKET_NAME is required (set it in .env or export it)"
  exit 1
fi

export STORAGE_MODE="gcs"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8080}"
# python -m pip install -r api/requirements.txt

cd api
python run_api.py
