from __future__ import annotations

import asyncio
from typing import Literal, Protocol, cast

from .wrapper import NlpStyle, apply_nlp, classify_locally, parse_llm_classification

NlpMode = Literal["off", "manual", "auto"]

NLP_CLASSIFICATION_SYSTEM_PROMPT = (
    "将中文文本分类为 NLP 改写模式。只输出 JSON，"
    '格式 {"style":"academic|general|long_blog"}。'
)
VALID_NLP_STYLES = {"academic", "general", "long_blog"}


class NlpClassifier(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str:
        ...


def normalize_nlp_style(style: str | None) -> NlpStyle:
    if style in VALID_NLP_STYLES:
        return cast(NlpStyle, style)
    return "academic"


async def choose_nlp_style(
    text: str,
    mode: NlpMode | str | None,
    requested_style: str | None,
    provider: NlpClassifier | None = None,
) -> NlpStyle:
    if mode == "auto":
        if provider is not None:
            try:
                raw = await provider.complete(
                    [
                        {"role": "system", "content": NLP_CLASSIFICATION_SYSTEM_PROMPT},
                        {"role": "user", "content": text[:1600]},
                    ]
                )
                parsed = parse_llm_classification(raw)
                if parsed is not None:
                    return parsed
            except Exception:
                pass
        return classify_locally(text)
    return normalize_nlp_style(requested_style)


async def rewrite_with_nlp_style(
    text: str,
    style: str | None,
    aggressive: bool = False,
    best_of_n: int = 10,
    seed: int | None = None,
) -> str:
    return await asyncio.to_thread(
        apply_nlp,
        text,
        normalize_nlp_style(style),
        aggressive,
        best_of_n,
        seed,
    )


async def rewrite_with_nlp(
    text: str,
    mode: NlpMode | str | None,
    requested_style: str | None,
    provider: NlpClassifier | None = None,
    aggressive: bool = False,
    best_of_n: int = 10,
    seed: int | None = None,
) -> tuple[str, NlpStyle]:
    style = await choose_nlp_style(text, mode, requested_style, provider)
    rewritten = await rewrite_with_nlp_style(text, style, aggressive, best_of_n, seed)
    return rewritten, style
