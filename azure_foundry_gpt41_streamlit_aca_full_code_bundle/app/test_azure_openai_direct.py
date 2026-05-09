from __future__ import annotations

import os

from azure_hosted_llm_generators import direct_azure_smoke_test, normalize_azure_base_url


def main() -> None:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_BASE_URL")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    key = os.getenv("AZURE_OPENAI_API_KEY", "")

    print("AZURE_OPENAI_BASE_URL:", normalize_azure_base_url(endpoint))
    print("AZURE_OPENAI_DEPLOYMENT:", deployment)
    print("AZURE_OPENAI_API_KEY loaded:", bool(key))
    print("AZURE_OPENAI_API_KEY length:", len(key))
    print(direct_azure_smoke_test())


if __name__ == "__main__":
    main()
