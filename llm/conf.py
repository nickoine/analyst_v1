# Built-in
from __future__ import annotations
import os

# Internal
from llm.dto import LLMConfig

def load_llm_settings() -> LLMConfig:
    """Load LLM routing settings from environment variables."""

    _api_key = os.getenv("OAI_API_KEY", "OPENROUTER_API_KEY").strip()
    if not _api_key:
        raise ValueError("API_KEY is required")

    model = (os.getenv("MODEL", "").strip() or None)
    base_url = (os.getenv("BASE_URL", "").strip() or None)
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", 100000))

    return LLMConfig(
        api_key=_api_key,
        model=model,
        base_url=base_url,
        max_output_tokens=max_output_tokens
    )
