#!/usr/bin/env bash
set -euo pipefail
RG="${RG:-rg_e222_tax_rag}"
APP_NAME="${APP_NAME:-taxrag-streamlit}"
az containerapp logs show --name "$APP_NAME" --resource-group "$RG" --follow
