# Revised Tax Facts RAG Benchmark Bundle

This bundle fixes the earlier design issue: the no-LLM path should not pretend to generate final answers. It is now a retrieval-only evidence baseline that abstains or reports evidence hits. Hosted LLM pipelines use the same retrieval evidence and generate answers with either OpenAI or Bedrock.

## Copy files into your project folder

```bash
cd ~/app_runner_demo/app_runner_tax_chatbot
# copy the files from this bundle into this folder
```

Required existing file:

```text
kpmg_tax_rag_v52_aws.py
```

Required artifact zip:

```text
~/app_runner_demo/runtime/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip
```

## Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy==2.4.3 pandas==2.2.3 tqdm boto3 python-dotenv sentence-transformers "transformers>=4.37,<5.6" accelerate scikit-learn scipy safetensors openai streamlit
```

## Set paths

```bash
export BASE_DIR=~/app_runner_demo/runtime
export ARTIFACT_ZIP="$BASE_DIR/kpmg_tax_rag_outputs_v52_corporate_50q-20260404T200240Z-1-001.zip"
export HF_HOME="$BASE_DIR/hf_cache"
mkdir -p "$BASE_DIR" "$HF_HOME"
```

## Run no-LLM evidence baseline

```bash
python run_taxfact_benchmark_v4.py --backend no_llm --questions taxfact_questions_v3.json --output-dir benchmark_outputs --label no_llm_taxfacts_v3
```

## Run OpenAI RAG

```bash
export OPENAI_API_KEY="YOUR_NEW_KEY"
export OPENAI_MODEL="gpt-4.1"
python run_taxfact_benchmark_v4.py --backend openai --questions taxfact_questions_v3.json --output-dir benchmark_outputs --label openai_taxfacts_v3
```

## Run Bedrock RAG

Use an inference profile ID/ARN if raw model ID fails.

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID="YOUR_INFERENCE_PROFILE_ID_OR_ARN"
python run_taxfact_benchmark_v4.py --backend bedrock --questions taxfact_questions_v3.json --output-dir benchmark_outputs --label bedrock_taxfacts_v3
```

## Compare

```bash
python compare_taxfact_runs_v2.py benchmark_outputs/no_llm_taxfacts_v3.json benchmark_outputs/openai_taxfacts_v3.json --output-dir benchmark_outputs --label no_llm_vs_openai_taxfacts_v3
```

## Streamlit

```bash
streamlit run streamlit_taxfact_app_v2.py --server.port=8080 --server.address=0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false --server.fileWatcherType none
```

Open SageMaker proxy URL, not `0.0.0.0`.
