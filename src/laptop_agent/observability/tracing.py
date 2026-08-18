"""LangSmith tracing.

Wiring choices worth stating explicitly:

* **Credentials come from the environment.** Nothing is hard-coded. When
  ``LANGSMITH_TRACING`` is false or no key is present, tracing is silently
  disabled and the graph runs unchanged.
* **Redaction happens inside the tracing client**, via ``hide_inputs`` /
  ``hide_outputs``. Every payload is filtered by the same
  :class:`~laptop_agent.security.redaction.SecretRedactor` the logs use, so a new
  pattern protects logs and traces together.
* **No hidden chain-of-thought is captured.** The configured model returns
  thinking with ``display: "omitted"`` by default, and nothing here requests
  summarised reasoning. What is traced is inputs needed for debugging, structured
  outputs, tool names, metadata, latency and errors.
* **The tracer is explicit, not ambient.** A ``LangChainTracer`` bound to our
  redacting client is passed in the run config rather than relying on global
  environment side effects, so a process that traces one graph does not
  accidentally trace everything.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import Settings, get_settings
from ..security.logging import get_logger
from ..security.redaction import SecretRedactor, build_redactor

_logger = get_logger("laptop_agent.tracing")

#: Top-level run name for a session (spec section 4).
SESSION_RUN_NAME = "laptop_agent_session"

#: Nested run names. These are the LangGraph node names, so the graph structure
#: and the trace structure cannot drift apart.
NODE_RUN_NAMES: tuple[str, ...] = (
    "input_guardrail",
    "requirements_analysis",
    "clarification_decision",
    "search_planning",
    "amazon_search",
    "flipkart_search",
    "offer_analysis",
    "pricing_calculation",
    "product_ranking",
    "recommendation_generation",
    "recommendation_validation",
)


class TracingSetup:
    """Holds the tracer and builds run configs."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self._settings = get_settings() if settings is None else settings
        self._redactor = redactor or build_redactor()
        self._tracer: Any | None = None
        self._client: Any | None = None
        if self._settings.tracing_enabled:
            self._tracer, self._client = self._build_tracer()

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    def _build_tracer(self) -> tuple[Any | None, Any | None]:
        settings = self._settings
        try:
            from langchain_core.tracers import LangChainTracer
            from langsmith import Client
        except ImportError:  # pragma: no cover - dependency is declared
            _logger.warning("tracing.disabled", extra={"reason": "langsmith_not_installed"})
            return None, None

        assert settings.langsmith_api_key is not None  # guarded by tracing_enabled

        # The SDK reads these for background submission of runs.
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)

        try:
            client = Client(
                api_url=settings.langsmith_endpoint,
                api_key=settings.langsmith_api_key.get_secret_value(),
                # Every traced payload passes through redaction before leaving
                # the process.
                hide_inputs=self._redactor.hide_inputs,
                hide_outputs=self._redactor.hide_outputs,
            )
            tracer = LangChainTracer(
                project_name=settings.langsmith_project, client=client
            )
        except Exception as exc:
            # Observability must never take the application down.
            _logger.warning(
                "tracing.disabled",
                extra={"reason": "client_init_failed", "error_type": type(exc).__name__},
            )
            return None, None

        _logger.info(
            "tracing.enabled",
            extra={"project": settings.langsmith_project, "endpoint": settings.langsmith_endpoint},
        )
        return tracer, client

    # ------------------------------------------------------------------

    def run_config(
        self,
        *,
        session_id: str,
        run_name: str = SESSION_RUN_NAME,
        extra_metadata: dict[str, Any] | None = None,
        extra_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the LangChain run config for one graph invocation."""
        settings = self._settings
        metadata: dict[str, Any] = {
            **settings.base_trace_metadata(),
            "session_id": session_id,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        tags = list(settings.base_trace_tags())
        if extra_tags:
            tags.extend(tag for tag in extra_tags if tag not in tags)

        config: dict[str, Any] = {
            "run_name": run_name,
            "tags": tags,
            "metadata": self._redactor.redact(metadata),
            # LangGraph needs a thread id when a checkpointer is attached; it is
            # also the natural correlation key in the trace UI.
            "configurable": {"thread_id": session_id},
        }
        if self._tracer is not None:
            config["callbacks"] = [self._tracer]
        return config

    def flush(self) -> None:
        """Best-effort flush of pending runs. Used on shutdown."""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:  # pragma: no cover - never fail on telemetry
            pass


def marketplace_tag(marketplace: str) -> str:
    """Consistent per-marketplace tag, e.g. ``marketplace:amazon``."""
    return f"marketplace:{marketplace}"
