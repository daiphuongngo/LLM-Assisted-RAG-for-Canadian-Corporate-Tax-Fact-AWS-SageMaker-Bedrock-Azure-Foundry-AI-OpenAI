#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../app"

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r ../requirements_azure_foundry.txt
python -m pip check || true

echo "Environment ready. Activate with:"
echo "cd $(pwd) && source .venv/bin/activate"
