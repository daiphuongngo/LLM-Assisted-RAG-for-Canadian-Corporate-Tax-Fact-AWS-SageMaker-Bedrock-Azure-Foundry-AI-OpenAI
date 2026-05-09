#!/usr/bin/env bash
set -euo pipefail

cd /workspace/app

export PORT="${PORT:-8000}"
export BASE_DIR="${BASE_DIR:-/workspace/runtime}"
export HF_HOME="${HF_HOME:-$BASE_DIR/hf_cache}"
export ARTIFACT_ZIP="${ARTIFACT_ZIP:-$(find "$BASE_DIR" -maxdepth 1 -name 'kpmg_tax_rag_outputs_v52_corporate_50q*.zip' | head -n 1)}"

export USE_LLM_PLANNER=false
export USE_LLM_ANSWER=false
export USE_LLM_VERIFIER=false
unset S3_BUCKET || true
unset S3_PREFIX || true
unset AWS_REGION || true
unset BEDROCK_MODEL_ID || true

if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
  echo "ERROR: AZURE_OPENAI_ENDPOINT is not set."
  exit 1
fi

if [[ -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: AZURE_OPENAI_API_KEY is not set. Use an Azure Container Apps secret reference."
  exit 1
fi

if [[ -z "${AZURE_OPENAI_DEPLOYMENT:-}" ]]; then
  echo "ERROR: AZURE_OPENAI_DEPLOYMENT is not set. Example: gpt-4.1"
  exit 1
fi

export AZURE_OPENAI_BASE_URL="${AZURE_OPENAI_BASE_URL:-${AZURE_OPENAI_ENDPOINT%/}/openai/v1/}"

mkdir -p "$BASE_DIR" "$HF_HOME" "$BASE_DIR/streamlit_logs"

echo "Starting Azure Tax RAG Streamlit app"
echo "PORT=$PORT"
echo "BASE_DIR=$BASE_DIR"
echo "ARTIFACT_ZIP=$ARTIFACT_ZIP"
echo "HF_HOME=$HF_HOME"
echo "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT"
echo "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT"
echo "Azure key loaded: yes"

exec streamlit run streamlit_taxfact_app_azure.py \
  --server.port="$PORT" \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.fileWatcherType=none \
  --server.headless=true
