from __future__ import annotations

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ProviderConfig
from .openai_provider import OpenAIProvider
from ..settings_service import RuntimeSettings


def get_provider(
    settings: RuntimeSettings,
    provider_name: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    active = provider_name or settings.provider
    resolved_model = model or settings.model_for(active)
    resolved_base_url = base_url or settings.base_url_for(active)
    resolved_key = api_key or settings.api_key_for(active)
    config = ProviderConfig(
        provider=active,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_key,
        request_url=settings.request_url_for(active),
    )
    if active == "openai":
        return OpenAIProvider(config)
    if active == "anthropic":
        return AnthropicProvider(config)
    raise ValueError(f"Unsupported provider: {active}")

