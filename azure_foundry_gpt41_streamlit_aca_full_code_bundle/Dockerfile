# Dockerfile for Azure Container Apps deployment of the Azure Foundry Tax RAG Streamlit app.
# Build context should be the project root: /home/dai/taxrag_azure_foundry

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BASE_DIR=/workspace/runtime \
    HF_HOME=/workspace/runtime/hf_cache \
    PORT=8000 \
    USE_LLM_PLANNER=false \
    USE_LLM_ANSWER=false \
    USE_LLM_VERIFIER=false

WORKDIR /workspace

# Minimal OS packages. git is useful for some Python packages and diagnostics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first to maximize build cache reuse.
COPY requirements_azure_foundry.txt /workspace/requirements_azure_foundry.txt

# Install CPU-only PyTorch first so sentence-transformers does not pull a larger CUDA-oriented wheel.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r /workspace/requirements_azure_foundry.txt \
    && python -m pip check || true

# Copy app code and the local retrieval artifact zip.
COPY app /workspace/app
COPY runtime/*.zip /workspace/runtime/

# Copy startup script.
COPY startup_aca.sh /workspace/startup_aca.sh
RUN chmod +x /workspace/startup_aca.sh \
    && mkdir -p /workspace/runtime/hf_cache /workspace/runtime/streamlit_logs

EXPOSE 8000

CMD ["/workspace/startup_aca.sh"]
