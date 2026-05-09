from __future__ import annotations

import json
import os

from tax_rag_v52_core import TaxRAGV52Core, make_settings_from_env, preflight_summary


def main() -> None:
    os.environ["USE_LLM_PLANNER"] = "false"
    os.environ["USE_LLM_ANSWER"] = "false"
    os.environ["USE_LLM_VERIFIER"] = "false"
    os.environ.pop("S3_BUCKET", None)
    os.environ.pop("S3_PREFIX", None)
    os.environ.pop("AWS_REGION", None)
    os.environ.pop("BEDROCK_MODEL_ID", None)

    settings = make_settings_from_env(
        base_dir=os.getenv("BASE_DIR"),
        bundle_zip_path=os.getenv("ARTIFACT_ZIP"),
    )

    settings.use_llm_planner = False
    settings.use_llm_answer = False
    settings.use_llm_verifier = False
    settings.s3_bucket = ""

    rag = TaxRAGV52Core(settings)
    rag.load_llm = lambda *args, **kwargs: (None, None)

    rag.extract_bundle_if_needed()

    summary = preflight_summary(rag.settings.artifacts_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
