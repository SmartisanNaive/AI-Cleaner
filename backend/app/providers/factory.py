from __future__ import annotations

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ProviderConfig
from .openai_provider import OpenAIProvider


def get_provider(config: ProviderConfig) -> LLMProvider:
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    raise ValueError(f"Unsupported provider: {config.provider}")
