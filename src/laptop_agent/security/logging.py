"""Structured JSON logging with redaction applied in the formatter.

Redaction sits in the formatter rather than at call sites so it cannot be
bypassed by a new log statement written later. Every record — message and
structured ``extra`` fields alike — passes through the redactor.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .redaction import SecretRedactor, build_redactor

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class RedactingJsonFormatter(logging.Formatter):
    """Emits one redacted JSON object per record."""

    def __init__(self, redactor: SecretRedactor) -> None:
        super().__init__()
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # Exception *type* and message only — no locals, which can hold secrets.
            exc_type, exc_value, _ = record.exc_info
            payload["error_type"] = getattr(exc_type, "__name__", str(exc_type))
            payload["error"] = str(exc_value)
        redacted = self._redactor.redact(payload)
        return json.dumps(redacted, default=str, ensure_ascii=False, sort_keys=True)


class RedactingTextFormatter(logging.Formatter):
    """Human-readable fallback for local development. Still redacted."""

    def __init__(self, redactor: SecretRedactor) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        return self._redactor.redact_text(super().format(record))


_configured = False


def configure_logging(
    *, level: str | None = None, as_json: bool | None = None
) -> SecretRedactor:
    """Install the redacting handler on the root logger. Idempotent."""
    global _configured

    from ..config import get_settings

    settings = get_settings()
    redactor = build_redactor()

    if _configured:
        return redactor

    resolved_level = (level or settings.log_level).upper()
    use_json = settings.log_json if as_json is None else as_json

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        RedactingJsonFormatter(redactor) if use_json else RedactingTextFormatter(redactor)
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # These libraries log request/response detail we do not want at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return redactor


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def reset_logging() -> None:
    """Test helper."""
    global _configured
    _configured = False
