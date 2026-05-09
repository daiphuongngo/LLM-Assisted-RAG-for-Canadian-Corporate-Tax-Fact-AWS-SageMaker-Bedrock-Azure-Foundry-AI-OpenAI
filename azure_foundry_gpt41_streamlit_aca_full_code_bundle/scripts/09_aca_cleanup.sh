#!/usr/bin/env bash
set -euo pipefail
RG="${RG:-rg_e222_tax_rag}"
APP_NAME="${APP_NAME:-taxrag-streamlit}"
ACA_ENV="${ACA_ENV:-taxrag-aca-env}"
ACR_NAME="${ACR_NAME:-}"

az containerapp delete --name "$APP_NAME" --resource-group "$RG" --yes || true
az containerapp env delete --name "$ACA_ENV" --resource-group "$RG" --yes || true
if [[ -n "$ACR_NAME" ]]; then
  az acr delete --name "$ACR_NAME" --resource-group "$RG" --yes || true
fi
