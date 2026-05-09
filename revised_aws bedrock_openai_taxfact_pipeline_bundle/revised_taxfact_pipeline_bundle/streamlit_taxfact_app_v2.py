from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from kpmg_tax_rag_v52_aws import KPMGTaxRAGV52AWS, make_settings_from_env
from hosted_llm_generators import answer_with_bedrock, answer_with_openai
from taxfact_retrieval_utils import make_augmented_query, safe_no_llm_answer, score_answer, select_evidence

st.set_page_config(page_title="Canadian Corporate Tax RAG", layout="wide")

BASE_DIR = Path(os.getenv("BASE_DIR", str(Path.cwd() / "runtime"))).expanduser()
ARTIFACT_ZIP = os.getenv("ARTIFACT_ZIP", "")
HF_HOME = os.getenv("HF_HOME", str(BASE_DIR / "hf_cache"))
os.environ.setdefault("HF_HOME", HF_HOME)

@st.cache_resource(show_spinner="Loading retrieval artifacts...")
def get_rag():
    settings = make_settings_from_env(base_dir=str(BASE_DIR), bundle_zip_path=ARTIFACT_ZIP)
    settings.use_llm_planner = False
    settings.use_llm_answer = False
    settings.use_llm_verifier = False
    rag = KPMGTaxRAGV52AWS(settings)
    rag.extract_bundle_if_needed()
    rag.sync_from_s3_if_needed()
    rag.load_artifacts()
    return rag

@st.cache_data
def load_questions(path: str = "taxfact_questions_v3.json"):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []

st.title("LLM-Assisted RAG for Canadian Corporate Tax Facts")
st.caption("Revised benchmark app: separates retrieval-only evidence baseline from hosted LLM answer generation.")

backend = st.sidebar.selectbox("Pipeline", ["no_llm", "openai", "bedrock"])
st.sidebar.write({
    "BASE_DIR": str(BASE_DIR),
    "ARTIFACT_ZIP": ARTIFACT_ZIP,
    "HF_HOME": HF_HOME,
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
    "BEDROCK_MODEL_ID": os.getenv("BEDROCK_MODEL_ID", ""),
})

questions = load_questions()
labels = [f"{q['id']} — {q['question']}" for q in questions]
choice = st.selectbox("Choose a benchmark question", labels if labels else ["No question file found"])
idx = labels.index(choice) if labels and choice in labels else 0
qmeta = questions[idx] if questions else {"question": "", "expected_answer": "", "expected_pages": [], "target_terms": []}
question = st.text_area("Question", value=qmeta.get("question", ""), height=100)

if st.button("Run pipeline"):
    t0 = time.time()
    rag = get_rag()
    retrieval_query = make_augmented_query({**qmeta, "question": question})
    raw = rag.answer_question(retrieval_query)
    evidence = select_evidence(raw, qmeta, max_items=6)

    if backend == "no_llm":
        model_result = safe_no_llm_answer(raw, qmeta)
        answer = model_result["answer"]
    elif backend == "openai":
        model_result = answer_with_openai(question, qmeta, evidence)
        answer = model_result.get("answer", "")
    else:
        model_result = answer_with_bedrock(question, qmeta, evidence)
        answer = model_result.get("answer", "")

    scores = score_answer(answer, qmeta.get("expected_answer", ""))
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
    (log_dir / f"result_{backend}_{qmeta.get('id','q')}_{stamp}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with (log_dir / "streamlit_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

    st.subheader("Answer")
    st.write(answer)
    st.subheader("Expected")
    st.write(qmeta.get("expected_answer"))
    st.subheader("Scores")
    st.json(scores)
    st.subheader("Selected evidence")
    for i, e in enumerate(evidence, start=1):
        with st.expander(f"Evidence {i} | page {e.get('printed_page')} | {e.get('section_title') or e.get('row_label')}"):
            st.write(str(e.get("text") or e.get("retrieval_text") or "")[:2000])
    st.success(f"Saved result to {log_dir}")
