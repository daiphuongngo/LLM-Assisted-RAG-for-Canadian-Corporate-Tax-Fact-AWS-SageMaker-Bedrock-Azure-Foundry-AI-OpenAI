"""
Azure-neutral alias around the preserved v5.2 retrieval core.

The original reconstructed file is named kpmg_tax_rag_v52_aws.py because it was
first patched for AWS portability. In this Azure Foundry version, the file is
used only as a local retrieval core. S3/AWS paths are disabled by the runner
scripts, and Azure OpenAI is used only for final answer generation.
"""
from __future__ import annotations

from kpmg_tax_rag_v52_aws import (  # noqa: F401
    KPMGTaxRAGV52AWS as TaxRAGV52Core,
    make_settings_from_env,
    preflight_summary,
)
