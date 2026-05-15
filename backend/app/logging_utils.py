from __future__ import annotations

import logging
import re


_SENSITIVE_PATTERNS = [
    re.compile(r"((?:api[_-]?key|authorization|base_url)\s*[:=]\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|authorization|base_url)['\"]?\s*[:=]\s*['\"])([^'\"]+)", re.IGNORECASE),
    re.compile(r"(Bearer\s+)([A-Za-z0-9._-]+)", re.IGNORECASE),
]


def _sanitize_text(text: str) -> str:
    sanitized = text
    for pattern in _SENSITIVE_PATTERNS:
        sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
    return sanitized


class SensitiveLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            return True
        record.msg = _sanitize_text(rendered)
        record.args = ()
        return True


def configure_logging_redaction() -> None:
    filter_instance = SensitiveLogFilter()
    for logger_name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "backend"):
        logger = logging.getLogger(logger_name)
        logger.addFilter(filter_instance)
        for handler in logger.handlers:
            handler.addFilter(filter_instance)
