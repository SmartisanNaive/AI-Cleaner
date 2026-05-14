from __future__ import annotations

from typing import AsyncIterator

from .base import ProviderConfig


class AnthropicProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config

    def _client(self):
        if not self.config.api_key:
            raise RuntimeError("Anthropic API Key 未配置。")
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=self.config.api_key, base_url=self.config.base_url, timeout=60.0)

    @staticmethod
    def _split_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        system_parts: list[str] = []
        user_messages: list[dict[str, str]] = []
        for message in messages:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                user_messages.append(message)
        return "\n\n".join(system_parts), user_messages

    async def complete(self, messages: list[dict[str, str]]) -> str:
        system, user_messages = self._split_messages(messages)
        client = self._client()
        response = await client.messages.create(
            model=self.config.model,
            max_tokens=4096,
            temperature=0.7,
            system=system or None,
            messages=user_messages,  # type: ignore[arg-type]
        )
        chunks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(chunks)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        system, user_messages = self._split_messages(messages)
        client = self._client()
        async with client.messages.stream(
            model=self.config.model,
            max_tokens=4096,
            temperature=0.7,
            system=system or None,
            messages=user_messages,  # type: ignore[arg-type]
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text

