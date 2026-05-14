from __future__ import annotations

from typing import AsyncIterator

from .base import ProviderConfig


class OpenAIProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config

    def _client(self):
        if not self.config.api_key:
            raise RuntimeError("OpenAI API Key 未配置。")
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url, timeout=60.0)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        client = self._client()
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        client = self._client()
        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta

