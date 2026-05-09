from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from taxfact_retrieval_utils import evidence_block


def build_prompt(question: str, qmeta: Dict[str, Any], evidence: List[Dict[str, Any]]) -> tuple[str, str]:
    """Build a grounded prompt for the final answer generator.

    The retrieval system already selected evidence. The hosted LLM should only
    synthesize an answer from that evidence and return machine-readable JSON.
    """
    system = (
        "You are a careful Canadian corporate tax facts QA assistant. "
        "Answer only from the provided evidence. Do not use outside knowledge. "
        "If the evidence does not support the answer, say 'Cannot determine from the retrieved evidence.' "
        "Return JSON only with keys: answer, confidence, cited_pages, reasoning. "
        "The confidence value must be one of: low, medium, high."
    )

    user = (
        f"Question: {question}\n"
        f"Section: {qmeta.get('section', '')}\n"
        f"Expected answer type: {qmeta.get('answer_kind', '')}\n"
        f"Expected printed page(s): {qmeta.get('expected_pages', [])}\n\n"
        f"Evidence:\n{evidence_block(evidence)}\n\n"
        "Instructions:\n"
        "1. For table values, return the exact value from the correct row and column.\n"
        "2. For deadline questions, include all relevant deadlines mentioned in the evidence.\n"
        "3. For notes questions, summarize only the relevant numbered note(s).\n"
        "4. Include cited page numbers in cited_pages.\n"
        "5. Return JSON only. Do not wrap the JSON in markdown.\n"
    )
    return system, user


def parse_jsonish(text: str) -> Dict[str, Any]:
    """Parse strict JSON when possible and fall back gracefully."""
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {
        "answer": text,
        "confidence": "unknown",
        "cited_pages": [],
        "reasoning": "unparsed_model_output",
    }



def stringify_answer_value(value: Any) -> str:
    """Return a safe plain-text answer for display and scoring.

    Azure OpenAI is instructed to return JSON, but the JSON value under
    ``answer`` can sometimes be a string, list, or dictionary. The Streamlit
    UI and benchmark scoring expect a string, so this helper normalizes the
    answer without losing information.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts: List[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                v_text = json.dumps(v, ensure_ascii=False)
            else:
                v_text = str(v)
            parts.append(f"{k}: {v_text}")
        return "; ".join(parts).strip()
    if isinstance(value, list):
        return " ".join(
            json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
            for x in value
        ).strip()
    return str(value).strip()

def response_output_text(response: Any) -> str:
    """Extract text from OpenAI Responses API response objects.

    Newer OpenAI SDK objects expose response.output_text. This fallback also
    handles model_dump() output for compatibility.
    """
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct)

    try:
        data = response.model_dump()
        parts: List[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    except Exception:
        pass

    return str(response)


def chat_completion_output_text(response: Any) -> str:
    """Extract text from Chat Completions API response objects."""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        pass

    try:
        data = response.model_dump()
        return data["choices"][0]["message"].get("content", "") or ""
    except Exception:
        pass

    return str(response)


def normalize_azure_base_url(raw: Optional[str]) -> str:
    """Return an Azure OpenAI v1 base_url.

    Accepts:
      https://RESOURCE.openai.azure.com
      https://RESOURCE.openai.azure.com/openai/v1/
      https://RESOURCE.services.ai.azure.com
      https://RESOURCE.services.ai.azure.com/openai/v1/

    Returns a URL ending in /openai/v1/.
    """
    if not raw:
        raise RuntimeError(
            "Missing Azure OpenAI endpoint. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT."
        )

    url = raw.strip().rstrip("/")

    if url.endswith("/openai/v1"):
        return url + "/"

    if url.endswith("/openai"):
        return url + "/v1/"

    return url + "/openai/v1/"


def _validate_azure_env() -> tuple[str, str, str]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    )
    raw_base_url = os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_API_KEY.")
    if not deployment:
        raise RuntimeError("Missing AZURE_OPENAI_DEPLOYMENT.")
    if not raw_base_url:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_BASE_URL.")

    return api_key, deployment, normalize_azure_base_url(raw_base_url)


def answer_with_azure_openai(question: str, qmeta: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a grounded answer with Azure AI Foundry / Azure OpenAI.

    Required environment variables:
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_BASE_URL
      AZURE_OPENAI_DEPLOYMENT

    Important:
      AZURE_OPENAI_DEPLOYMENT is the Azure deployment name created in Foundry,
      not necessarily the raw base model name.
    """
    from openai import OpenAI

    api_key, deployment, base_url = _validate_azure_env()
    client = OpenAI(api_key=api_key, base_url=base_url)
    system, user = build_prompt(question, qmeta, evidence)

    # Preferred modern Azure OpenAI path: Responses API.
    try:
        response = client.responses.create(
            model=deployment,
            instructions=system,
            input=user,
            temperature=0,
            max_output_tokens=700,
        )
        raw = response_output_text(response)
        api_used = "responses"
    except Exception as responses_exc:
        # Fallback path if a deployment/region supports chat completions but not responses.
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=700,
        )
        raw = chat_completion_output_text(response)
        api_used = f"chat_completions_fallback_after_responses_error: {type(responses_exc).__name__}"

    parsed = parse_jsonish(raw)

    # Downstream display and scoring expect a plain string answer.
    parsed["answer"] = stringify_answer_value(parsed.get("answer", ""))

    parsed["raw_model_output"] = raw
    parsed["model"] = deployment
    parsed["backend"] = "azure_openai"
    parsed["azure_base_url"] = base_url
    parsed["api_used"] = api_used
    return parsed


def direct_azure_smoke_test() -> str:
    """Small direct model call used by scripts/debugging."""
    from openai import OpenAI

    api_key, deployment, base_url = _validate_azure_env()
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model=deployment,
        input='Return JSON only: {"status":"ok","message":"Azure OpenAI test passed"}',
        temperature=0,
        max_output_tokens=100,
    )
    return response_output_text(response)
