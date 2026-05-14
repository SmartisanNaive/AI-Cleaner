from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ProviderName = Literal["openai", "anthropic"]
HistoryProviderName = Literal["openai", "anthropic", "local"]
PlatformName = Literal["weipu", "paperyy", "paperpass", "zhuque"]
NlpMode = Literal["off", "manual", "auto"]
NlpStyle = Literal["academic", "general", "long_blog"]


class SettingsPayload(BaseModel):
    provider: ProviderName = "openai"
    openai_model: str = "gpt-5.4"
    anthropic_model: str = "claude-4-6-sonnet"
    openai_base_url: str | None = None
    anthropic_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    stream: bool = True
    nlp_enabled: bool = False
    nlp_mode: NlpMode = "manual"
    nlp_style: NlpStyle = "academic"


class SettingsView(BaseModel):
    provider: ProviderName
    openai_model: str
    anthropic_model: str
    openai_base_url: str
    anthropic_base_url: str
    openai_api_key_set: bool
    anthropic_api_key_set: bool
    openai_api_key_source: str
    anthropic_api_key_source: str
    stream: bool
    nlp_enabled: bool
    nlp_mode: NlpMode
    nlp_style: NlpStyle
    openai_request_url: str
    anthropic_request_url: str
    warnings: list[str] = Field(default_factory=list)


class SettingsTestRequest(BaseModel):
    provider: ProviderName | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class SettingsTestResponse(BaseModel):
    ok: bool
    provider: ProviderName
    request_url: str
    latency_ms: int
    response_preview: str | None = None
    error: str | None = None


class RewriteRequest(BaseModel):
    text: str
    platform: PlatformName = "weipu"
    iterations: int = Field(default=1, ge=1, le=5)
    provider: ProviderName | None = None
    model: str | None = None
    stream: bool = False
    nlp_enabled: bool | None = None
    nlp_mode: NlpMode | None = None
    nlp_style: NlpStyle | None = None
    nlp_aggressive: bool = False
    nlp_best_of_n: int = Field(default=10, ge=0, le=20)
    nlp_seed: int | None = None


class NlpRewriteRequest(BaseModel):
    text: str
    platform: PlatformName = "weipu"
    nlp_mode: NlpMode = "manual"
    nlp_style: NlpStyle = "academic"
    aggressive: bool = False
    best_of_n: int = Field(default=10, ge=0, le=20)
    seed: int | None = None


class DiffSpan(BaseModel):
    kind: Literal["equal", "insert", "delete", "replace"]
    original: str = ""
    revised: str = ""


class RewriteResponse(BaseModel):
    id: int
    original_text: str
    rewritten_text: str
    raw_output: str
    platform: PlatformName
    provider: HistoryProviderName
    model: str
    iterations: int
    warnings: list[str]
    nlp_applied: bool
    nlp_style: NlpStyle | None = None
    diff: list[DiffSpan]
    created_at: datetime


class HistoryItem(BaseModel):
    id: int
    platform: PlatformName
    provider: HistoryProviderName
    model: str
    original_preview: str
    rewritten_preview: str
    created_at: datetime
