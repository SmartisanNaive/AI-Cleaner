from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    request_url: str


class LLMProvider(Protocol):
    config: ProviderConfig

    async def complete(self, messages: list[dict[str, str]]) -> str:
        ...

    async def stream(self, messages: list[dict[str, str]], request_id: str | None = None) -> AsyncIterator[str]:
        ...

