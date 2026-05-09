#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../app"
source .venv/bin/activate

pkill -f streamlit || true

streamlit run streamlit_taxfact_app_azure.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.fileWatcherType none
