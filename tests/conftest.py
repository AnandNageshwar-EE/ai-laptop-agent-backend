"""Shared fixtures.

Tests run entirely offline: no API key, no network, no LangSmith. That is a
property of the design (``LLM_MODE=offline`` uses the deterministic reasoner),
not a mocking layer bolted on for tests.
"""

from __future__ import annotations

import pytest

from laptop_agent.audit import CollectingAuditSink
from laptop_agent.cache import InMemoryCacheProvider
from laptop_agent.config import Settings, reset_settings_cache
from laptop_agent.domain import Bank, Currency, LaptopRequirements, Money, PurchaseProfile, UseCase
from laptop_agent.marketplace.registry import build_registry
from laptop_agent.session import InMemorySessionStore


#: Environment that must not leak into tests from a developer's local .env.
#: Without this, a machine configured with MARKETPLACE_SOURCE=serpapi makes the
#: suite hit a live API — slow, non-deterministic, and it spends real credits.
_ISOLATED_ENV = {
    "MARKETPLACE_SOURCE": "fixtures",
    "SERPAPI_KEY": "",
    "LLM_MODE": "offline",
    "ANTHROPIC_API_KEY": "",
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_API_KEY": "",
    "ENVIRONMENT": "test",
}


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every test onto offline, fixture-backed, untraced configuration."""
    for name, value in _ISOLATED_ENV.items():
        monkeypatch.setenv(name, value)
    # Settings are memoised, so the cache must be dropped either side.
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_mode="offline",
        marketplace_source="fixtures",
        serpapi_key=None,
        langsmith_tracing=False,
        environment="test",
        log_json=True,
    )


@pytest.fixture
def audit() -> CollectingAuditSink:
    return CollectingAuditSink()


@pytest.fixture
def cache() -> InMemoryCacheProvider:
    return InMemoryCacheProvider()


@pytest.fixture
def registry(cache: InMemoryCacheProvider, settings: Settings):
    return build_registry(cache=cache, settings=settings)


@pytest.fixture
def sessions() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def inr() -> type[Currency]:
    return Currency


@pytest.fixture
def dev_requirements() -> LaptopRequirements:
    return LaptopRequirements(
        use_case=UseCase.SOFTWARE_DEVELOPMENT,
        budget_max=Money(amount="90000", currency=Currency.INR),
        min_ram_gb=16,
        mandatory_fields=["budget_max", "min_ram_gb"],
    )


@pytest.fixture
def hdfc_profile() -> PurchaseProfile:
    return PurchaseProfile(eligible_banks=[Bank.HDFC])


@pytest.fixture
def empty_profile() -> PurchaseProfile:
    return PurchaseProfile()
