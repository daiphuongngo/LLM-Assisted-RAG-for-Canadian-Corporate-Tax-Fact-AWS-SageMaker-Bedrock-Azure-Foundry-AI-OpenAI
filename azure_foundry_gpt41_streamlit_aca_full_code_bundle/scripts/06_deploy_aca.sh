#!/usr/bin/env bash
set -euo pipefail

# Run from project root: /home/dai/taxrag_azure_foundry
# Prerequisite: source scripts/02_export_env_template.sh so Azure OpenAI variables are loaded.

RG="${RG:-rg_e222_tax_rag}"
LOCATION="${LOCATION:-eastus}"
ACA_ENV="${ACA_ENV:-taxrag-aca-env}"
APP_NAME="${APP_NAME:-taxrag-streamlit}"
IMAGE_NAME="${IMAGE_NAME:-taxrag-streamlit}"
TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"
ACR_NAME="${ACR_NAME:-acrtaxrag$RANDOM}"

if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" || -z "${AZURE_OPENAI_API_KEY:-}" || -z "${AZURE_OPENAI_DEPLOYMENT:-}" ]]; then
  echo "ERROR: Azure OpenAI env vars missing. Run: source scripts/02_export_env_template.sh"
  exit 1
fi

if [[ ! -f "Dockerfile" ]]; then
  echo "ERROR: Dockerfile not found. Run this script from /home/dai/taxrag_azure_foundry."
  exit 1
fi

if ! compgen -G "runtime/kpmg_tax_rag_outputs_v52_corporate_50q*.zip" > /dev/null; then
  echo "ERROR: RAG artifact zip not found in runtime/."
  echo "Expected: runtime/kpmg_tax_rag_outputs_v52_corporate_50q*.zip"
  exit 1
fi

az extension add --name containerapp --upgrade >/dev/null
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az group create --name "$RG" --location "$LOCATION" >/dev/null

if ! az acr show --name "$ACR_NAME" --resource-group "$RG" >/dev/null 2>&1; then
  az acr create --resource-group "$RG" --name "$ACR_NAME" --sku Basic --admin-enabled false >/dev/null
fi

ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --resource-group "$RG" --query loginServer -o tsv)"
FULL_IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG"

echo "Building image in Azure Container Registry: $FULL_IMAGE"
az acr build --registry "$ACR_NAME" --image "$IMAGE_NAME:$TAG" .

if ! az containerapp env show --name "$ACA_ENV" --resource-group "$RG" >/dev/null 2>&1; then
  az containerapp env create --name "$ACA_ENV" --resource-group "$RG" --location "$LOCATION" >/dev/null
fi

ARTIFACT_FILE="$(basename "$(ls runtime/kpmg_tax_rag_outputs_v52_corporate_50q*.zip | head -n 1)")"

echo "Creating/updating Azure Container App: $APP_NAME"
if az containerapp show --name "$APP_NAME" --resource-group "$RG" >/dev/null 2>&1; then
  az containerapp secret set \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --secrets azure-openai-api-key="$AZURE_OPENAI_API_KEY" >/dev/null

  az containerapp update \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --image "$FULL_IMAGE" \
    --set-env-vars \
      PORT=8000 \
      BASE_DIR=/workspace/runtime \
      ARTIFACT_ZIP="/workspace/runtime/$ARTIFACT_FILE" \
      HF_HOME=/workspace/runtime/hf_cache \
      USE_LLM_PLANNER=false \
      USE_LLM_ANSWER=false \
      USE_LLM_VERIFIER=false \
      AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
      AZURE_OPENAI_BASE_URL="${AZURE_OPENAI_ENDPOINT%/}/openai/v1/" \
      AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
      AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key >/dev/null
else
  az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --environment "$ACA_ENV" \
    --image "$FULL_IMAGE" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-identity system \
    --target-port 8000 \
    --ingress external \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas 0 \
    --max-replicas 1 \
    --secrets azure-openai-api-key="$AZURE_OPENAI_API_KEY" \
    --env-vars \
      PORT=8000 \
      BASE_DIR=/workspace/runtime \
      ARTIFACT_ZIP="/workspace/runtime/$ARTIFACT_FILE" \
      HF_HOME=/workspace/runtime/hf_cache \
      USE_LLM_PLANNER=false \
      USE_LLM_ANSWER=false \
      USE_LLM_VERIFIER=false \
      AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
      AZURE_OPENAI_BASE_URL="${AZURE_OPENAI_ENDPOINT%/}/openai/v1/" \
      AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
      AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key >/dev/null
fi

FQDN="$(az containerapp show --name "$APP_NAME" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)"

echo ""
echo "Deployment complete."
echo "Resource group: $RG"
echo "ACR name:       $ACR_NAME"
echo "Image:          $FULL_IMAGE"
echo "App name:       $APP_NAME"
echo "URL:            https://$FQDN"
echo ""
echo "View logs:"
echo "az containerapp logs show --name $APP_NAME --resource-group $RG --follow"
