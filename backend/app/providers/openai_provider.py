from __future__ import annotations

import logging
from typing import AsyncIterator

from .base import ProviderConfig


logger = logging.getLogger(__name__)


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
        try:
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("OpenAI complete failed: model=%s base_url=%s", self.config.model, self.config.base_url)
            raise

    async def stream(self, messages: list[dict[str, str]], request_id: str | None = None) -> AsyncIterator[str]:
        client = self._client()
        try:
            stream = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                stream=True,
                extra_headers={"X-Request-ID": request_id} if request_id else None,
            )
            async for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta
        except Exception:
            logger.exception("OpenAI stream failed: request_id=%s model=%s base_url=%s", request_id, self.config.model, self.config.base_url)
            raise

