from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from kpmg_tax_rag_v52_aws import KPMGTaxRAGV52AWS, make_settings_from_env
from taxfact_retrieval_utils import (
    make_augmented_query,
    page_hit,
    safe_no_llm_answer,
    score_answer,
    select_evidence,
)
from hosted_llm_generators import answer_with_bedrock, answer_with_openai


def build_retrieval_rag() -> KPMGTaxRAGV52AWS:
    settings = make_settings_from_env(base_dir=os.getenv("BASE_DIR"), bundle_zip_path=os.getenv("ARTIFACT_ZIP"))
    # Important: retrieval-only base. Do not load local Qwen for benchmark evidence retrieval.
    settings.use_llm_planner = False
    settings.use_llm_answer = False
    settings.use_llm_verifier = False
    rag = KPMGTaxRAGV52AWS(settings)
    rag.extract_bundle_if_needed()
    rag.sync_from_s3_if_needed()
    rag.load_artifacts()
    return rag


def extract_generated_answer(model_result: Dict[str, Any]) -> str:
    return str(model_result.get("answer") or "").strip()


def run_one(rag: KPMGTaxRAGV52AWS, backend: str, qmeta: Dict[str, Any]) -> Dict[str, Any]:
    retrieval_query = make_augmented_query(qmeta)
    t0 = time.time()
    raw = rag.answer_question(retrieval_query)
    retrieval_seconds = time.time() - t0
    evidence = select_evidence(raw, qmeta, max_items=6)

    if backend == "no_llm":
        model_result = safe_no_llm_answer(raw, qmeta)
        answer = model_result["answer"]
    elif backend == "openai":
        t1 = time.time()
        model_result = answer_with_openai(qmeta["question"], qmeta, evidence)
        model_result["generation_seconds"] = round(time.time() - t1, 3)
        answer = extract_generated_answer(model_result)
    elif backend == "bedrock":
        t1 = time.time()
        model_result = answer_with_bedrock(qmeta["question"], qmeta, evidence)
        model_result["generation_seconds"] = round(time.time() - t1, 3)
        answer = extract_generated_answer(model_result)
    else:
        raise ValueError(f"Unsupported backend for this benchmark: {backend}")

    scores = score_answer(answer, qmeta.get("expected_answer", ""))
    return {
        "id": qmeta["id"],
        "backend": backend,
        "question": qmeta["question"],
        "section": qmeta.get("section"),
        "answer_kind": qmeta.get("answer_kind"),
        "expected_pages": qmeta.get("expected_pages", []),
        "expected_answer": qmeta.get("expected_answer", ""),
        "answer": answer,
        "page_hit": page_hit(raw, qmeta.get("expected_pages", [])),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "total_seconds": round(time.time() - t0, 3),
        **scores,
        "model_result": model_result,
        "selected_evidence": [
            {
                "printed_page": e.get("printed_page"),
                "section_title": e.get("section_title"),
                "row_label": e.get("row_label"),
                "content_type": e.get("content_type"),
                "text_preview": str(e.get("text") or e.get("retrieval_text") or "")[:800],
            }
            for e in evidence
        ],
        "raw_retrieval_result": raw,
    }


def summarize(rows: List[Dict[str, Any]], label: str, backend: str) -> Dict[str, Any]:
    n = len(rows)
    return {
        "label": label,
        "backend": backend,
        "n": n,
        "page_hit_rate": sum(bool(r["page_hit"]) for r in rows) / n if n else 0,
        "exact_match_rate": sum(bool(r["exact_match"]) for r in rows) / n if n else 0,
        "contains_expected_rate": sum(bool(r["contains_expected"]) for r in rows) / n if n else 0,
        "value_hit_rate": sum(bool(r["value_hit"]) for r in rows) / n if n else 0,
        "mean_total_seconds": round(sum(float(r["total_seconds"]) for r in rows) / n, 3) if n else 0,
    }


def write_outputs(output: Dict[str, Any], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    txt_path = output_dir / f"{label}.txt"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(output["summary"], indent=2, ensure_ascii=False))
        f.write("\n\n")
        for r in output["rows"]:
            f.write("=" * 100 + "\n")
            f.write(f"{r['id']} | {r['backend']} | {r['section']} | {r['answer_kind']}\n")
            f.write(f"Question: {r['question']}\n\n")
            f.write(f"Expected:\n{r['expected_answer']}\n\n")
            f.write(f"Answer:\n{r['answer']}\n\n")
            f.write(f"Scores: page_hit={r['page_hit']} exact={r['exact_match']} contains={r['contains_expected']} value_hit={r['value_hit']}\n")
            f.write(f"Seconds: retrieval={r['retrieval_seconds']} total={r['total_seconds']}\n")
            f.write("Selected evidence:\n")
            for e in r["selected_evidence"]:
                f.write(f"- p.{e.get('printed_page')} | {e.get('content_type')} | {e.get('section_title') or e.get('row_label')}\n")
                f.write(f"  {e.get('text_preview')}\n")
            f.write("\n")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["no_llm", "openai", "bedrock"])
    ap.add_argument("--questions", default="taxfact_questions_v3.json")
    ap.add_argument("--output-dir", default="benchmark_outputs")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rag = build_retrieval_rag()
    rows = []
    for q in questions:
        print(f"Running {q['id']} {args.backend}: {q['question']}")
        rows.append(run_one(rag, args.backend, q))
    output = {"summary": summarize(rows, args.label, args.backend), "rows": rows}
    write_outputs(output, Path(args.output_dir), args.label)


if __name__ == "__main__":
    main()
