"""Application configuration.

Every credential is read from the environment. Nothing is hard-coded, and no
secret value is ever placed in a log line, a trace payload or a repr — secrets
are held as ``SecretStr`` so an accidental f-string yields ``**********``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "bedrock"]
LLMMode = Literal["live", "offline"]
#: Where marketplace listings come from. ``fixtures`` needs no credentials and
#: runs offline; ``serpapi`` fetches live listings and real prices.
MarketplaceSource = Literal["fixtures", "serpapi"]

# Volatile data must never be cached for long. Section 6 of the NFR spec.
MAX_VOLATILE_CACHE_TTL_SECONDS = 300


class Settings(BaseSettings):
    """Runtime settings, sourced from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- application identity (used for tracing metadata) -----
    application: str = "ai-laptop-agent"
    environment: str = "local"
    graph_version: str = "v1"

    # ----- HTTP API -----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Streamlit dev server origins. Explicit allowlist, never "*".
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )

    # ----- LLM -----
    llm_provider: LLMProvider = "anthropic"
    # `offline` runs deterministic rule-based reasoning so the graph, guardrails
    # and validators are fully exercisable without a live API key.
    llm_mode: LLMMode = "offline"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 4096
    llm_timeout_seconds: float = 60.0
    # Structured-output retries. Spec section 2: retry once, then fail gracefully.
    llm_structured_retries: int = 1

    anthropic_api_key: SecretStr | None = None
    aws_region: str = "us-east-1"

    # ----- marketplace data source -----
    marketplace_source: MarketplaceSource = "fixtures"
    serpapi_key: SecretStr | None = None
    serpapi_timeout_seconds: float = 20.0
    serpapi_country: str = "in"

    # ----- prompt caching -----
    prompt_caching_enabled: bool = True
    # Provider-side ephemeral cache TTL: "5m" (default) or "1h".
    prompt_cache_ttl: Literal["5m", "1h"] = "5m"

    # ----- product / offer caching (volatile: short TTL only) -----
    product_cache_ttl_seconds: int = 300  # 5 minutes
    offer_cache_ttl_seconds: int = 120  # 1-5 minutes

    # ----- guardrails -----
    max_input_chars: int = 2_000
    min_input_chars: int = 2
    max_search_results: int = 20
    recommendation_max_repair_attempts: int = 2

    # ----- session state (in-memory; swappable, see session/store.py) -----
    session_ttl_seconds: int = 1_800
    max_sessions: int = 1_000

    # ----- LangSmith -----
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "ai-laptop-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ----- logging -----
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("product_cache_ttl_seconds", "offer_cache_ttl_seconds")
    @classmethod
    def _cap_volatile_ttl(cls, value: int) -> int:
        """Price data is volatile — refuse to configure a long-lived price cache."""
        if value < 0:
            raise ValueError("cache TTL cannot be negative")
        if value > MAX_VOLATILE_CACHE_TTL_SECONDS:
            raise ValueError(
                "price/offer data is volatile: TTL must be <= "
                f"{MAX_VOLATILE_CACHE_TTL_SECONDS} seconds (5 minutes)"
            )
        return value

    @property
    def live_marketplace_enabled(self) -> bool:
        """True only when a live source is selected *and* usable."""
        return self.marketplace_source == "serpapi" and self.serpapi_key is not None

    @property
    def effective_marketplace_source(self) -> str:
        """What will actually be used, after falling back on a missing key."""
        return "serpapi" if self.live_marketplace_enabled else "fixtures"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langsmith_tracing and self.langsmith_api_key)

    @property
    def traced_model_name(self) -> str:
        return self.llm_model if self.llm_mode == "live" else "offline-deterministic"

    def base_trace_metadata(self) -> dict[str, str]:
        """Static metadata attached to every LangSmith run (spec section 4.1)."""
        return {
            "application": self.application,
            "environment": self.environment,
            "graph_version": self.graph_version,
            "model": self.traced_model_name,
            "llm_provider": self.llm_provider,
            "marketplace_source": self.effective_marketplace_source,
        }

    def base_trace_tags(self) -> list[str]:
        """Static tags attached to every LangSmith run (spec section 4.2)."""
        return [
            "agent:laptop-shopping",
            f"graph:{self.graph_version}",
            f"environment:{self.environment}",
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Test helper — drop the memoised settings instance."""
    get_settings.cache_clear()
