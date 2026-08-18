"""Structured LLM invocation with validation and a single bounded retry.

The contract, per spec section 2:

1. Ask for structured output constrained by a JSON schema.
2. Validate the response against the Pydantic model.
3. On failure, retry **once** with a different constraining mechanism.
4. If it still fails, fail gracefully — never fall back to parsing free text.

The retry deliberately switches strategy rather than repeating the same request:
``json_schema`` uses the provider's response-format constraint, and the fallback
uses tool-calling, which constrains the shape differently. Repeating an identical
failing request is rarely useful.

Usage metadata (tokens, cache reads, retry count) is returned alongside the
parsed model so a node can attach it to the trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from ..security.logging import get_logger

TModel = TypeVar("TModel", bound=BaseModel)

_logger = get_logger("laptop_agent.llm")


class StructuredOutputError(RuntimeError):
    """The model could not produce a valid response within the retry budget."""

    def __init__(self, schema: str, attempts: int, detail: str) -> None:
        self.schema = schema
        self.attempts = attempts
        self.detail = detail
        super().__init__(f"{schema}: invalid structured output after {attempts} attempt(s)")


@dataclass(slots=True)
class InvocationStats:
    """Observability for one structured call. Contains no prompt content."""

    schema: str
    attempts: int = 0
    retries: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    methods_used: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_tokens > 0

    def as_metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "attempts": self.attempts,
            "retry_count": self.retries,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_tokens,
            "cache_creation_input_tokens": self.cache_creation_tokens,
            "prompt_cache_hit": self.cache_hit,
            "structured_output_methods": ",".join(self.methods_used),
        }


#: Strategies in order. The second is the retry.
_STRATEGIES: tuple[str, ...] = ("json_schema", "function_calling")


class StructuredLLM:
    """Invokes a chat model and returns a validated Pydantic model."""

    def __init__(self, chat_model: Any, *, max_retries: int = 1) -> None:
        self._model = chat_model
        self._max_retries = max(0, max_retries)

    def invoke(
        self,
        schema: type[TModel],
        messages: list[BaseMessage],
        *,
        config: dict[str, Any] | None = None,
    ) -> tuple[TModel, InvocationStats]:
        stats = InvocationStats(schema=schema.__name__)
        started = time.perf_counter()
        last_detail = "unknown"

        strategies = _STRATEGIES[: 1 + self._max_retries]
        for index, method in enumerate(strategies):
            stats.attempts += 1
            stats.methods_used.append(method)
            if index > 0:
                stats.retries += 1

            try:
                parsed, raw = self._attempt(schema, messages, method, config)
            except ValidationError as exc:
                last_detail = _summarise(exc)
                stats.validation_errors.append(last_detail)
                _logger.warning(
                    "structured_output.validation_failed",
                    extra={
                        "schema": schema.__name__,
                        "method": method,
                        "attempt": stats.attempts,
                        "error_summary": last_detail,
                    },
                )
                continue
            except Exception as exc:  # provider/transport failure
                last_detail = f"{type(exc).__name__}: {str(exc)[:160]}"
                stats.validation_errors.append(last_detail)
                _logger.warning(
                    "structured_output.call_failed",
                    extra={
                        "schema": schema.__name__,
                        "method": method,
                        "attempt": stats.attempts,
                    },
                )
                continue

            self._record_usage(stats, raw)
            stats.latency_ms = (time.perf_counter() - started) * 1000
            return parsed, stats

        stats.latency_ms = (time.perf_counter() - started) * 1000
        # No free-text fallback. The caller degrades gracefully instead.
        raise StructuredOutputError(schema.__name__, stats.attempts, last_detail)

    # ------------------------------------------------------------------

    def _attempt(
        self,
        schema: type[TModel],
        messages: list[BaseMessage],
        method: str,
        config: dict[str, Any] | None,
    ) -> tuple[TModel, Any]:
        runnable = self._model.with_structured_output(
            schema, method=method, include_raw=True
        )
        result = runnable.invoke(messages, config=config or {})

        # include_raw=True yields {"raw", "parsed", "parsing_error"}.
        if isinstance(result, dict):
            error = result.get("parsing_error")
            parsed = result.get("parsed")
            raw = result.get("raw")
            if error is not None:
                raise error if isinstance(error, Exception) else ValueError(str(error))
            if parsed is None:
                raise ValueError("model returned no parsable structured output")
            # Re-validate even though the integration already parsed: this class
            # is the boundary, and it does not trust its dependencies to have
            # applied our validators.
            return schema.model_validate(
                parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
            ), raw

        return schema.model_validate(
            result.model_dump() if isinstance(result, BaseModel) else result
        ), None

    @staticmethod
    def _record_usage(stats: InvocationStats, raw: Any) -> None:
        """Pull token and cache counters out of the response.

        ``cache_read_input_tokens`` is the number that proves prompt caching is
        actually working; if it stays zero across repeated calls, the stable
        prefix has drifted.
        """
        usage = getattr(raw, "usage_metadata", None) or {}
        if not isinstance(usage, dict):
            return
        stats.input_tokens = int(usage.get("input_tokens") or 0)
        stats.output_tokens = int(usage.get("output_tokens") or 0)
        details = usage.get("input_token_details") or {}
        if isinstance(details, dict):
            stats.cache_read_tokens = int(details.get("cache_read") or 0)
            stats.cache_creation_tokens = int(details.get("cache_creation") or 0)


def _summarise(exc: ValidationError) -> str:
    """First few field errors, without echoing the model's raw output."""
    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(piece) for piece in error.get("loc", ())) or "<root>"
        parts.append(f"{location}:{error.get('type', 'invalid')}")
    return "; ".join(parts)
