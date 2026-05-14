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
from .database import get_setting, set_setting
from .schemas import SettingsPayload, SettingsView
from .security import decrypt_secret, encrypt_secret


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _warn_urls(openai_base_url: str, anthropic_base_url: str) -> list[str]:
    warnings: list[str] = []
    if not openai_base_url.rstrip("/").endswith("/v1"):
        warnings.append("OpenAI Chat Completions 通常需要 base URL 以 /v1 结尾。")
    if "/v1/v1" in _join_url(anthropic_base_url, ANTHROPIC_MESSAGES_PATH):
        warnings.append("Anthropic 请求 URL 中出现 /v1/v1，请检查 base URL 是否重复包含 /v1。")
    return warnings


@dataclass(frozen=True)
class RuntimeSettings:
    provider: str
    openai_model: str
    anthropic_model: str
    openai_base_url: str
    anthropic_base_url: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    openai_api_key_source: str
    anthropic_api_key_source: str
    stream: bool
    nlp_enabled: bool
    nlp_mode: str
    nlp_style: str

    def model_for(self, provider: str | None = None) -> str:
        active = provider or self.provider
        return self.openai_model if active == "openai" else self.anthropic_model

    def api_key_for(self, provider: str) -> str | None:
        return self.openai_api_key if provider == "openai" else self.anthropic_api_key

    def base_url_for(self, provider: str) -> str:
        return self.openai_base_url if provider == "openai" else self.anthropic_base_url

    def request_url_for(self, provider: str) -> str:
        if provider == "openai":
            return _join_url(self.openai_base_url, OPENAI_CHAT_COMPLETIONS_PATH)
        return _join_url(self.anthropic_base_url, ANTHROPIC_MESSAGES_PATH)


def _get_bool(key: str, default: bool) -> bool:
    raw = get_setting(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def load_settings() -> RuntimeSettings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    stored_openai_key = decrypt_secret(get_setting("openai_api_key"))
    stored_anthropic_key = decrypt_secret(get_setting("anthropic_api_key"))
    env_openai_key = os.getenv("OPENAI_API_KEY")
    env_anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    return RuntimeSettings(
        provider=get_setting("provider") or "openai",
        openai_model=os.getenv("OPENAI_MODEL") or get_setting("openai_model") or DEFAULT_OPENAI_MODEL,
        anthropic_model=os.getenv("ANTHROPIC_MODEL")
        or get_setting("anthropic_model")
        or DEFAULT_ANTHROPIC_MODEL,
        openai_base_url=os.getenv("OPENAI_BASE_URL")
        or get_setting("openai_base_url")
        or DEFAULT_OPENAI_BASE_URL,
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL")
        or get_setting("anthropic_base_url")
        or DEFAULT_ANTHROPIC_BASE_URL,
        openai_api_key=env_openai_key or stored_openai_key,
        anthropic_api_key=env_anthropic_key or stored_anthropic_key,
        openai_api_key_source="env"
        if env_openai_key
        else ("encrypted" if stored_openai_key else "missing"),
        anthropic_api_key_source="env"
        if env_anthropic_key
        else ("encrypted" if stored_anthropic_key else "missing"),
        stream=_get_bool("stream", True),
        nlp_enabled=_get_bool("nlp_enabled", False),
        nlp_mode=get_setting("nlp_mode") or "manual",
        nlp_style=get_setting("nlp_style") or "academic",
    )


def save_settings(payload: SettingsPayload) -> RuntimeSettings:
    set_setting("provider", payload.provider)
    set_setting("openai_model", payload.openai_model)
    set_setting("anthropic_model", payload.anthropic_model)
    set_setting("openai_base_url", payload.openai_base_url or DEFAULT_OPENAI_BASE_URL)
    set_setting("anthropic_base_url", payload.anthropic_base_url or DEFAULT_ANTHROPIC_BASE_URL)
    set_setting("stream", "true" if payload.stream else "false")
    set_setting("nlp_enabled", "true" if payload.nlp_enabled else "false")
    set_setting("nlp_mode", payload.nlp_mode)
    set_setting("nlp_style", payload.nlp_style)
    if payload.openai_api_key is not None and payload.openai_api_key.strip():
        set_setting("openai_api_key", encrypt_secret(payload.openai_api_key.strip()) or "")
    if payload.anthropic_api_key is not None and payload.anthropic_api_key.strip():
        set_setting("anthropic_api_key", encrypt_secret(payload.anthropic_api_key.strip()) or "")
    return load_settings()


def settings_view(settings: RuntimeSettings | None = None) -> SettingsView:
    settings = settings or load_settings()
    return SettingsView(
        provider=settings.provider,  # type: ignore[arg-type]
        openai_model=settings.openai_model,
        anthropic_model=settings.anthropic_model,
        openai_base_url=settings.openai_base_url,
        anthropic_base_url=settings.anthropic_base_url,
        openai_api_key_set=bool(settings.openai_api_key),
        anthropic_api_key_set=bool(settings.anthropic_api_key),
        openai_api_key_source=settings.openai_api_key_source,
        anthropic_api_key_source=settings.anthropic_api_key_source,
        stream=settings.stream,
        nlp_enabled=settings.nlp_enabled,
        nlp_mode=settings.nlp_mode,  # type: ignore[arg-type]
        nlp_style=settings.nlp_style,  # type: ignore[arg-type]
        openai_request_url=settings.request_url_for("openai"),
        anthropic_request_url=settings.request_url_for("anthropic"),
        warnings=_warn_urls(settings.openai_base_url, settings.anthropic_base_url),
    )
