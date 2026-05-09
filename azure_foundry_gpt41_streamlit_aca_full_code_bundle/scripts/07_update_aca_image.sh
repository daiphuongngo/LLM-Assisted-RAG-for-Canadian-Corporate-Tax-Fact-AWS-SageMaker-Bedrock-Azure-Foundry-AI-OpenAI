#!/usr/bin/env bash
set -euo pipefail

# Run from project root after editing code.
RG="${RG:-rg_e222_tax_rag}"
APP_NAME="${APP_NAME:-taxrag-streamlit}"
IMAGE_NAME="${IMAGE_NAME:-taxrag-streamlit}"
TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"
ACR_NAME="${ACR_NAME:?Set ACR_NAME to the registry created by 06_deploy_aca.sh}"

ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --resource-group "$RG" --query loginServer -o tsv)"
FULL_IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG"

az acr build --registry "$ACR_NAME" --image "$IMAGE_NAME:$TAG" .
az containerapp update --name "$APP_NAME" --resource-group "$RG" --image "$FULL_IMAGE" >/dev/null

FQDN="$(az containerapp show --name "$APP_NAME" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo "Updated app: https://$FQDN"
