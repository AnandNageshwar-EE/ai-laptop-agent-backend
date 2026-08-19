"""Model construction.

Claude Opus 5 via ``langchain-anthropic`` is the default, chosen because it
supports the two features this design depends on: native prompt caching
(``cache_control`` on content blocks) and JSON-schema structured output.

Notes on parameters that are easy to get wrong on this model family:

* ``temperature`` / ``top_p`` / ``top_k`` are **removed** on Claude Opus 5 and
  are rejected with a 400. They are deliberately never set here.
* Thinking is on by default and the raw chain of thought is never returned
  (``display`` defaults to ``omitted``), which matches the requirement not to
  capture hidden reasoning in traces.
* The model is fixed for the life of a conversation. Switching models
  invalidates every prompt-cache entry, since caches are model-scoped.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings


class LLMUnavailableError(RuntimeError):
    """Raised when a live model is requested but cannot be constructed."""


def build_chat_model(settings: Settings | None = None) -> Any:
    """Construct the chat model for the configured provider.

    Returns a LangChain ``BaseChatModel``. Typed ``Any`` so importing this module
    does not require the provider package to be installed.
    """
    resolved = get_settings() if settings is None else settings

    if resolved.llm_provider == "anthropic":
        return _build_anthropic(resolved)
    if resolved.llm_provider == "bedrock":
        return _build_bedrock(resolved)
    if resolved.llm_provider == "openrouter":
        return _build_openrouter(resolved)
    raise LLMUnavailableError(f"unsupported llm_provider: {resolved.llm_provider!r}")


def _build_anthropic(settings: Settings) -> Any:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise LLMUnavailableError(
            "langchain-anthropic is not installed; install it or set LLM_MODE=offline"
        ) from exc

    if settings.anthropic_api_key is None:
        raise LLMUnavailableError(
            "ANTHROPIC_API_KEY is not set; set it or set LLM_MODE=offline"
        )

    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        # Retries are handled by the structured-output layer so a retry is
        # observable in the trace rather than hidden inside the client.
        max_retries=0,
        api_key=settings.anthropic_api_key,
        # temperature/top_p/top_k intentionally omitted — rejected on this model.
    )


def _build_bedrock(settings: Settings) -> Any:
    """Amazon Bedrock path.

    Prompt caching on Bedrock is supported but the cache-write/read accounting
    and the minimum cacheable prefix are the provider's, not identical to the
    first-party API. Verify ``usage_metadata`` before assuming cache hits.
    """
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:
        raise LLMUnavailableError(
            "llm_provider='bedrock' requires the langchain-aws package. "
            "Install it, or use llm_provider='anthropic'."
        ) from exc

    return ChatBedrockConverse(
        model=f"anthropic.{settings.llm_model}",
        region_name=settings.aws_region,
        max_tokens=settings.llm_max_tokens,
    )


def _build_openrouter(settings: Settings) -> Any:
    """OpenRouter, an OpenAI-compatible gateway.

    OpenRouter exposes **only** the OpenAI chat-completions schema — there is no
    Anthropic-native ``/v1/messages`` endpoint — so this path uses ``ChatOpenAI``
    with an overridden base URL rather than ``ChatAnthropic``.

    Prompt caching still works for models whose upstream provider supports it
    (Anthropic, Gemini, OpenAI): the ``cache_control`` markers this application
    already puts on its system blocks are forwarded. Models that cannot cache
    ignore the markers harmlessly, and
    :attr:`Settings.provider_prompt_caching_supported` reports that honestly
    instead of implying tokens are being saved.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMUnavailableError(
            "llm_provider='openrouter' requires the langchain-openai package."
        ) from exc

    if settings.openrouter_api_key is None:
        raise LLMUnavailableError(
            "OPENROUTER_API_KEY is not set; set it or set LLM_MODE=offline"
        )

    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        # Retries are owned by the structured-output layer so a retry is visible
        # in the trace rather than hidden in the client.
        max_retries=0,
        default_headers={
            "HTTP-Referer": settings.openrouter_site_url,
            "X-Title": settings.openrouter_app_name,
        },
        # Ask OpenRouter to report cache accounting back to us.
        extra_body={"usage": {"include": True}},
    )


def describe_model(settings: Settings | None = None) -> dict[str, str]:
    """Model identity for trace metadata. Contains no credentials."""
    resolved = get_settings() if settings is None else settings
    return {
        "llm_mode": resolved.llm_mode,
        "llm_provider": resolved.llm_provider,
        "model": resolved.traced_model_name,
    }
