from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from azure_hosted_llm_generators import answer_with_azure_openai, stringify_answer_value
from tax_rag_v52_core import TaxRAGV52Core, make_settings_from_env
from taxfact_retrieval_utils import (
    make_augmented_query,
    safe_no_llm_answer,
    score_answer,
    select_evidence,
)

st.set_page_config(page_title="Azure Foundry Canadian Corporate Tax RAG", layout="wide")

BASE_DIR = Path(os.getenv("BASE_DIR", str(Path.cwd() / "runtime"))).expanduser()
ARTIFACT_ZIP = os.getenv("ARTIFACT_ZIP", "")
HF_HOME = os.getenv("HF_HOME", str(BASE_DIR / "hf_cache"))
os.environ.setdefault("HF_HOME", HF_HOME)


def disable_non_azure_paths() -> None:
    os.environ["USE_LLM_PLANNER"] = "false"
    os.environ["USE_LLM_ANSWER"] = "false"
    os.environ["USE_LLM_VERIFIER"] = "false"
    os.environ.pop("S3_BUCKET", None)
    os.environ.pop("S3_PREFIX", None)
    os.environ.pop("AWS_REGION", None)
    os.environ.pop("BEDROCK_MODEL_ID", None)


@st.cache_resource(show_spinner="Loading local retrieval artifacts...")
def get_rag():
    disable_non_azure_paths()

    settings = make_settings_from_env(
        base_dir=str(BASE_DIR),
        bundle_zip_path=ARTIFACT_ZIP,
    )

    settings.use_llm_planner = False
    settings.use_llm_answer = False
    settings.use_llm_verifier = False
    settings.s3_bucket = ""

    rag = TaxRAGV52Core(settings)

    # Safety guard: Streamlit must not load local Qwen.
    rag.load_llm = lambda *args, **kwargs: (None, None)

    rag.extract_bundle_if_needed()
    rag.load_artifacts()

    rag.settings.use_llm_planner = False
    rag.settings.use_llm_answer = False
    rag.settings.use_llm_verifier = False
    rag.settings.s3_bucket = ""
    rag.load_llm = lambda *args, **kwargs: (None, None)

    return rag


@st.cache_data
def load_questions(path: str = "taxfact_questions_v3.json"):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


st.title("Azure AI Foundry RAG for Canadian Corporate Tax Facts")
st.caption(
    "Azure-only version: local v5.2 retrieval artifact + Azure OpenAI final answer generation."
)

backend = st.sidebar.selectbox(
    "Pipeline",
    ["no_llm", "azure_openai"],
)

st.sidebar.write(
    {
        "BASE_DIR": str(BASE_DIR),
        "ARTIFACT_ZIP": ARTIFACT_ZIP,
        "HF_HOME": HF_HOME,
        "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        "AZURE_OPENAI_ENDPOINT_SET": bool(
            os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_BASE_URL")
        ),
        "AZURE_OPENAI_KEY_LOADED": bool(os.getenv("AZURE_OPENAI_API_KEY")),
        "LOCAL_QWEN_DISABLED": True,
        "AWS_DISABLED": True,
    }
)

questions = load_questions()
labels = [f"{q['id']} — {q['question']}" for q in questions]

choice = st.selectbox(
    "Choose a benchmark question",
    labels if labels else ["No question file found"],
)

idx = labels.index(choice) if labels and choice in labels else 0
qmeta = questions[idx] if questions else {
    "question": "",
    "expected_answer": "",
    "expected_pages": [],
    "target_terms": [],
}

question = st.text_area(
    "Question",
    value=qmeta.get("question", ""),
    height=100,
)

if st.button("Run pipeline"):
    t0 = time.time()

    rag = get_rag()

    retrieval_query = make_augmented_query({**qmeta, "question": question})
    raw = rag.answer_question(retrieval_query)
    evidence = select_evidence(raw, qmeta, max_items=6)

    if backend == "no_llm":
        model_result = safe_no_llm_answer(raw, qmeta)
        answer = model_result["answer"]

    elif backend == "azure_openai":
        model_result = answer_with_azure_openai(question, qmeta, evidence)
        answer = stringify_answer_value(model_result.get("answer", ""))
        model_result["answer"] = answer

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    scores = score_answer(str(answer), str(qmeta.get("expected_answer", "")))

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": backend,
        "question_id": qmeta.get("id"),
        "question": question,
        "expected_answer": qmeta.get("expected_answer"),
        "answer": answer,
        "scores": scores,
        "elapsed_seconds": round(time.time() - t0, 3),
        "model_result": model_result,
        "selected_evidence": evidence,
    }

    log_dir = BASE_DIR / "streamlit_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"result_{backend}_{qmeta.get('id', 'q')}_{stamp}.json"
    log_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    with (log_dir / "streamlit_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Expected")
    st.write(qmeta.get("expected_answer"))

    st.subheader("Scores")
    st.json(scores)

    st.subheader("Model result")
    st.json(
        {
            k: v
            for k, v in model_result.items()
            if k not in {"raw_model_output"}
        }
    )

    with st.expander("Raw model output"):
        st.write(model_result.get("raw_model_output", ""))

    st.subheader("Selected evidence")
    for i, e in enumerate(evidence, start=1):
        with st.expander(
            f"Evidence {i} | page {e.get('printed_page')} | "
            f"{e.get('section_title') or e.get('row_label')}"
        ):
            st.write(str(e.get("text") or e.get("retrieval_text") or "")[:2000])

    st.success(f"Saved result to {log_path}")
