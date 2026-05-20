from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ProviderName = Literal["openai", "anthropic"]
HistoryProviderName = Literal["openai", "anthropic", "local"]
PlatformName = Literal["weipu", "paperyy", "paperpass", "zhuque"]
NlpMode = Literal["off", "manual", "auto"]
NlpStyle = Literal["academic", "general", "long_blog"]


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
    base_url: str | None = None
    api_key: str | None = None
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
