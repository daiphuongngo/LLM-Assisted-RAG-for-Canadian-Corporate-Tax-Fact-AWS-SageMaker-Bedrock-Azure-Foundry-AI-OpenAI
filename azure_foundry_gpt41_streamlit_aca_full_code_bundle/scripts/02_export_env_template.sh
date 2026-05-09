#!/usr/bin/env bash
# Source this file after editing the placeholders.
# Example: source ../scripts/02_export_env_template.sh

export BASE_DIR="$HOME/taxrag_azure_foundry/runtime"
export ARTIFACT_ZIP="$BASE_DIR/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip"
export HF_HOME="$BASE_DIR/hf_cache"
mkdir -p "$BASE_DIR" "$HF_HOME"

export USE_LLM_PLANNER=false
export USE_LLM_ANSWER=false
export USE_LLM_VERIFIER=false

unset S3_BUCKET
unset S3_PREFIX
unset AWS_REGION
unset BEDROCK_MODEL_ID

export AZURE_OPENAI_ENDPOINT="https://YOUR-AZURE-OPENAI-RESOURCE.openai.azure.com"
export AZURE_OPENAI_BASE_URL="${AZURE_OPENAI_ENDPOINT%/}/openai/v1/"
export AZURE_OPENAI_API_KEY="PASTE_YOUR_AZURE_OPENAI_KEY_HERE"
export AZURE_OPENAI_DEPLOYMENT="PASTE_YOUR_DEPLOYMENT_NAME_HERE"

echo "BASE_DIR=$BASE_DIR"
echo "ARTIFACT_ZIP=$ARTIFACT_ZIP"
echo "HF_HOME=$HF_HOME"
echo "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT"
echo "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT"
echo "Azure key loaded: $([ -n "$AZURE_OPENAI_API_KEY" ] && echo yes || echo no)"
