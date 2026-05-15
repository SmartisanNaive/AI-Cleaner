from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .constants import (
    ANTHROPIC_MESSAGES_PATH,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    OPENAI_CHAT_COMPLETIONS_PATH,
    PROJECT_ROOT,
)
from .providers.base import ProviderConfig


@dataclass(frozen=True)
class RuntimeDefaults:
    openai_model: str
    anthropic_model: str
    openai_base_url: str
    anthropic_base_url: str
    openai_api_key: str | None
    anthropic_api_key: str | None

    def model_for(self, provider: str) -> str:
        return self.openai_model if provider == "openai" else self.anthropic_model

    def base_url_for(self, provider: str) -> str:
        return self.openai_base_url if provider == "openai" else self.anthropic_base_url

    def api_key_for(self, provider: str) -> str | None:
        return self.openai_api_key if provider == "openai" else self.anthropic_api_key


def load_runtime_defaults() -> RuntimeDefaults:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return RuntimeDefaults(
        openai_model=os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        anthropic_model=os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def preview_request_url(provider: str, base_url: str | None = None) -> str:
    defaults = load_runtime_defaults()
    resolved_base_url = (base_url or defaults.base_url_for(provider)).rstrip("/")
    if provider == "openai":
        return resolved_base_url + OPENAI_CHAT_COMPLETIONS_PATH
    return resolved_base_url + ANTHROPIC_MESSAGES_PATH


def resolve_provider_config(
    provider_name: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderConfig:
    defaults = load_runtime_defaults()
    active = provider_name or "openai"
    resolved_model = (model or defaults.model_for(active)).strip()
    resolved_base_url = (base_url or defaults.base_url_for(active)).strip()
    resolved_api_key = api_key.strip() if api_key and api_key.strip() else defaults.api_key_for(active)
    return ProviderConfig(
        provider=active,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        request_url=preview_request_url(active, resolved_base_url),
    )
