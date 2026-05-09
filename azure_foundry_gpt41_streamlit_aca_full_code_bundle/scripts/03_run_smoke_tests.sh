#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../app"
source .venv/bin/activate

python test_azure_openai_direct.py
python preflight_azure_taxrag.py

python - <<'PY'
import json
from pathlib import Path
questions = json.loads(Path("taxfact_questions_v3.json").read_text(encoding="utf-8"))
Path("taxfact_questions_v3_smoke.json").write_text(
    json.dumps(questions[:3], indent=2, ensure_ascii=False), encoding="utf-8"
)
print("Created taxfact_questions_v3_smoke.json with", len(questions[:3]), "questions.")
PY

python run_taxfact_benchmark_azure.py \
  --backend no_llm \
  --questions taxfact_questions_v3_smoke.json \
  --output-dir benchmark_outputs \
  --label smoke_no_llm_taxfacts_v3

python run_taxfact_benchmark_azure.py \
  --backend azure_openai \
  --questions taxfact_questions_v3_smoke.json \
  --output-dir benchmark_outputs \
  --label smoke_azure_openai_taxfacts_v3
