# Azure Foundry / Azure OpenAI GPT-4.1 Tax RAG - Streamlit + Azure Container Apps

This bundle packages the Azure-only version of the Canadian corporate tax RAG project.
It keeps the local v5.2 retrieval artifact, disables local Qwen/AWS/Bedrock paths, and uses Azure OpenAI GPT-4.1 in Azure AI Foundry for final answer generation.

## Architecture

```text
Streamlit UI on Azure Container Apps
        |
        v
Local v5.2 retrieval artifact in /workspace/runtime
        |
        v
Selected evidence from retrieved tax-fact chunks
        |
        v
Azure OpenAI GPT-4.1 deployment through /openai/v1/
        |
        v
Grounded JSON answer + cited pages + reasoning
```

## Required external artifact

Before running locally or deploying to Azure Container Apps, place this artifact zip in `runtime/`:

```text
runtime/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip
```

The artifact zip is not bundled here because it is a project/runtime data artifact rather than a source-code file.

## Main files

```text
app/kpmg_tax_rag_v52_aws.py              # preserved v5.2 retrieval core; AWS disabled by Azure runners
app/tax_rag_v52_core.py                  # Azure-neutral alias to retrieval core
app/taxfact_retrieval_utils.py           # retrieval evidence utilities and scoring
app/azure_hosted_llm_generators.py       # Azure OpenAI GPT-4.1 generator
app/streamlit_taxfact_app_azure.py       # Streamlit UI
app/run_taxfact_benchmark_azure.py       # no_llm and azure_openai benchmark runner
app/preflight_azure_taxrag.py            # artifact preflight
app/test_azure_openai_direct.py          # direct Azure OpenAI smoke test
app/compare_taxfact_runs_v2.py           # comparison utility
app/taxfact_questions_v3.json            # 10-question benchmark; Q6 expected answer corrected
Dockerfile                               # Azure Container Apps image build
startup_aca.sh                           # Streamlit startup script inside container
scripts/06_deploy_aca.sh                 # first deployment to Azure Container Apps
scripts/07_update_aca_image.sh           # rebuild/update after code changes
scripts/08_aca_logs.sh                   # logs
scripts/09_aca_cleanup.sh                # cleanup
```

## Security

Do not commit a real Azure OpenAI API key. Use:

```bash
scripts/02_export_env_template.sh
```

for local shell testing, and Azure Container Apps secrets for deployment. If a key was shown in a screenshot, regenerate it before final submission or public demo.

## Local Azure compute test

```bash
cd ~/taxrag_azure_foundry
bash scripts/01_setup_environment.sh
nano scripts/02_export_env_template.sh
source scripts/02_export_env_template.sh
bash scripts/03_run_smoke_tests.sh
bash scripts/04_run_full_benchmark.sh
```

## Streamlit local/Cloud Shell test

```bash
cd ~/taxrag_azure_foundry
source scripts/02_export_env_template.sh
source app/.venv/bin/activate
cd app
streamlit run streamlit_taxfact_app_azure.py \
  --server.port=8000 \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.fileWatcherType=none \
  --server.headless=true
```

Cloud Shell Web Preview can be unreliable for Streamlit. Azure Container Apps is the recommended deployed UI.

## Azure Container Apps deployment

```bash
cd ~/taxrag_azure_foundry
source scripts/02_export_env_template.sh

export RG="rg_e222_tax_rag"
export LOCATION="eastus"
export ACA_ENV="taxrag-aca-env"
export APP_NAME="taxrag-streamlit"
export ACR_NAME="acrtaxrag$(date +%m%d%H%M)$RANDOM"

bash scripts/06_deploy_aca.sh
```

The script builds the image using Azure Container Registry, stores the Azure OpenAI key as a Container Apps secret, deploys the app, and prints the public HTTPS URL.

## Updating after a code change

```bash
cd ~/taxrag_azure_foundry
source scripts/02_export_env_template.sh

export RG="rg_e222_tax_rag"
export APP_NAME="taxrag-streamlit"
export ACR_NAME="<ACR_NAME_FROM_FIRST_DEPLOY>"

bash scripts/07_update_aca_image.sh
```

## Keeping the demo warm

```bash
az containerapp update \
  --name taxrag-streamlit \
  --resource-group rg_e222_tax_rag \
  --min-replicas 1 \
  --max-replicas 1
```

After the demo:

```bash
az containerapp update \
  --name taxrag-streamlit \
  --resource-group rg_e222_tax_rag \
  --min-replicas 0 \
  --max-replicas 1
```

## Q6 correction

The original Q6 expected answer was accidentally copied from the expenditure-limit note. The corrected Q6 expected answer is the R&D ITC note about applying/refunding/carrying ITCs, T2 Schedule 31, CRA filing within 12 months of the return filing due date, T661 forms, and deduction of ITCs from the R&D expenditure pool. The model's generated Q6 answer matched the corrected source evidence, so the earlier automatic metric understated performance.
