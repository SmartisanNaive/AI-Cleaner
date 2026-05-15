from __future__ import annotations

import time
from datetime import datetime, timezone

from .schemas import DiffSpan, NlpStyle, RewriteResponse


def next_response_id() -> int:
    return int(time.time() * 1000)


def build_rewrite_response(
    *,
    original_text: str,
    rewritten_text: str,
    raw_output: str,
    platform: str,
    provider: str,
    model: str,
    iterations: int,
    warnings: list[str],
    nlp_applied: bool,
    diff: list[dict[str, str]] | list[DiffSpan],
    nlp_style: str | NlpStyle | None = None,
    response_id: int | None = None,
    created_at: datetime | None = None,
) -> RewriteResponse:
    created_at = created_at or datetime.now(timezone.utc)
    return RewriteResponse(
        id=response_id or next_response_id(),
        original_text=original_text,
        rewritten_text=rewritten_text,
        raw_output=raw_output,
        platform=platform,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        model=model,
        iterations=iterations,
        warnings=warnings,
        nlp_applied=nlp_applied,
        nlp_style=nlp_style,  # type: ignore[arg-type]
        diff=diff,  # type: ignore[arg-type]
        created_at=created_at,
    )
