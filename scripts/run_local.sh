#!/usr/bin/env bash
set -euo pipefail

# cd to repo root
cd "$(dirname "$0")/.."

# load .env if present
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

export STORAGE_MODE="${STORAGE_MODE:-local}"
export ASSETS_DIR="${ASSETS_DIR:-../assets}"
export API_KEY="${API_KEY:-demo-key-123}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8080}"

# install deps if needed (optional; comment out if you prefer manual install)
# python -m pip install -r api/requirements.txt

cd api
python run_api.py
