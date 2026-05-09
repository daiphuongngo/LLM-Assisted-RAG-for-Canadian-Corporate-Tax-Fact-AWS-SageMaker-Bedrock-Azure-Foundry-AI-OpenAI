# Azure AI Foundry / Azure OpenAI RAG Pipeline for Canadian Corporate Tax Facts

This package is an Azure-only version of the Canadian corporate tax RAG project. It keeps the local v5.2 retrieval artifact and uses Azure OpenAI through Azure AI Foundry for final answer generation.

## What this package does

- Loads the preserved v5.2 tax RAG artifact locally.
- Disables local Qwen loading.
- Disables AWS/S3/Bedrock paths.
- Runs a no-LLM retrieval-only baseline.
- Runs an Azure OpenAI answer-generation backend.
- Runs a Streamlit app on Azure compute.

## Required external artifact

Place the artifact zip below in `~/taxrag_azure_foundry/runtime/`:

```text
kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip
```

This artifact is not included in this code package if it is larger than the course upload limit.

## Azure AI Foundry setup

Create an Azure OpenAI deployment in Azure AI Foundry and record:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT
```

The deployment name is the name you gave the model deployment in Azure AI Foundry.

## Folder setup on Azure compute

Recommended location:

```bash
mkdir -p ~/taxrag_azure_foundry
cd ~/taxrag_azure_foundry
unzip azure_foundry_openai_taxrag_code.zip
mkdir -p runtime runtime/hf_cache
cp /path/to/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip runtime/
```

## Install

```bash
cd ~/taxrag_azure_foundry
bash scripts/01_setup_environment.sh
```

## Environment variables

Edit and source:

```bash
nano scripts/02_export_env_template.sh
source scripts/02_export_env_template.sh
```

Make sure these are real values:

```bash
export AZURE_OPENAI_ENDPOINT="https://YOUR-AZURE-OPENAI-RESOURCE.openai.azure.com"
export AZURE_OPENAI_API_KEY="PASTE_YOUR_AZURE_OPENAI_KEY_HERE"
export AZURE_OPENAI_DEPLOYMENT="PASTE_YOUR_DEPLOYMENT_NAME_HERE"
```

## Smoke tests

```bash
cd ~/taxrag_azure_foundry
source scripts/02_export_env_template.sh
bash scripts/03_run_smoke_tests.sh
```

## Full benchmark

```bash
cd ~/taxrag_azure_foundry
source scripts/02_export_env_template.sh
bash scripts/04_run_full_benchmark.sh
```

Outputs are written to:

```text
app/benchmark_outputs/
```

## Streamlit

```bash
cd ~/taxrag_azure_foundry
source scripts/02_export_env_template.sh
bash scripts/05_run_streamlit.sh
```

Open the forwarded port for `8501` in Azure AI Foundry / Azure ML / VS Code.

## Main files

```text
app/kpmg_tax_rag_v52_aws.py              # preserved retrieval core; AWS disabled in Azure runner
app/tax_rag_v52_core.py                  # Azure-neutral alias
app/taxfact_retrieval_utils.py           # evidence selection and scoring utilities
app/azure_hosted_llm_generators.py       # Azure OpenAI generator
app/preflight_azure_taxrag.py            # artifact preflight
app/test_azure_openai_direct.py          # direct Azure OpenAI test
app/run_taxfact_benchmark_azure.py       # benchmark runner
app/streamlit_taxfact_app_azure.py       # Streamlit demo
app/compare_taxfact_runs_v2.py           # comparison utility
app/taxfact_questions_v3.json            # revised benchmark questions
```

## Security note

Do not commit API keys to GitHub. Use environment variables or Azure Key Vault for a real deployment.
