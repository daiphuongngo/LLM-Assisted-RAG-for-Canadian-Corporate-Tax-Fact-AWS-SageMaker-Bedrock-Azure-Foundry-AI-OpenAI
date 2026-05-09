#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../app"
source .venv/bin/activate

python run_taxfact_benchmark_azure.py \
  --backend no_llm \
  --questions taxfact_questions_v3.json \
  --output-dir benchmark_outputs \
  --label no_llm_taxfacts_v3

python run_taxfact_benchmark_azure.py \
  --backend azure_openai \
  --questions taxfact_questions_v3.json \
  --output-dir benchmark_outputs \
  --label azure_openai_taxfacts_v3

python compare_taxfact_runs_v2.py \
  benchmark_outputs/no_llm_taxfacts_v3.json \
  benchmark_outputs/azure_openai_taxfacts_v3.json \
  --output-dir benchmark_outputs \
  --label no_llm_vs_azure_openai_taxfacts_v3

ls -lh benchmark_outputs
