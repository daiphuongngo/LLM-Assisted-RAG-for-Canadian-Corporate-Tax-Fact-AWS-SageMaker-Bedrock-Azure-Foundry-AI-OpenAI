from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from taxfact_retrieval_utils import evidence_block


def build_prompt(question: str, qmeta: Dict[str, Any], evidence: List[Dict[str, Any]]) -> tuple[str, str]:
    system = (
        "You are a careful Canadian corporate tax facts QA assistant. "
        "Answer only from the evidence. Do not use outside knowledge. "
        "If the evidence does not support the answer, say 'Cannot determine from the retrieved evidence.' "
        "Return JSON only with keys: answer, confidence, cited_pages, reasoning."
    )
    user = (
        f"Question: {question}\n"
        f"Section: {qmeta.get('section', '')}\n"
        f"Expected answer type: {qmeta.get('answer_kind', '')}\n"
        f"Expected printed page(s): {qmeta.get('expected_pages', [])}\n\n"
        f"Evidence:\n{evidence_block(evidence)}\n\n"
        "Instructions:\n"
        "1. For table values, return the exact value from the correct row and column.\n"
        "2. For notes questions, summarize only the relevant numbered note(s).\n"
        "3. Include cited page numbers in cited_pages.\n"
        "4. JSON only."
    )
    return system, user


def parse_jsonish(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {"answer": text, "confidence": "unknown", "cited_pages": [], "reasoning": "unparsed_model_output"}


def answer_with_openai(question: str, qmeta: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    system, user = build_prompt(question, qmeta, evidence)
    response = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        temperature=0,
        max_output_tokens=700,
    )
    raw = getattr(response, "output_text", None) or str(response)
    parsed = parse_jsonish(raw)
    parsed["raw_model_output"] = raw
    parsed["model"] = model
    parsed["backend"] = "openai"
    return parsed


def answer_with_bedrock(question: str, qmeta: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    import boto3
    from botocore.config import Config
    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0")
    client = boto3.client("bedrock-runtime", region_name=region, config=Config(read_timeout=3600))
    system, user = build_prompt(question, qmeta, evidence)
    response = client.converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": 700, "temperature": 0},
    )
    raw = "".join(block.get("text", "") for block in response["output"]["message"]["content"])
    parsed = parse_jsonish(raw)
    parsed["raw_model_output"] = raw
    parsed["model"] = model_id
    parsed["backend"] = "bedrock"
    return parsed
