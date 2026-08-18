"""Marketplace client registry.

Providers are registered once at startup. The graph asks the registry for a
client by :class:`~laptop_agent.domain.enums.Marketplace` — an enum, so there is
no string lookup a model could influence, and no way to name a provider that
does not exist.
"""

from __future__ import annotations

from ..cache.base import CacheProvider
from ..config import Settings, get_settings
from ..domain.enums import Marketplace
from ..security.logging import get_logger
from .base import MarketplaceClient, ProductTransport
from .clients import AmazonClient, FlipkartClient

_logger = get_logger("laptop_agent.marketplace")


class MarketplaceRegistry:
    def __init__(self, clients: dict[Marketplace, MarketplaceClient]) -> None:
        self._clients = clients

    def get(self, marketplace: Marketplace) -> MarketplaceClient:
        client = self._clients.get(marketplace)
        if client is None:
            raise KeyError(f"no client registered for {marketplace}")
        return client

    @property
    def marketplaces(self) -> list[Marketplace]:
        # Deterministic order keeps trace and result ordering stable.
        return sorted(self._clients, key=lambda market: market.value)

    def __len__(self) -> int:
        return len(self._clients)


def build_transport(settings: Settings) -> ProductTransport | None:
    """Live transport, or ``None`` to use the bundled fixtures.

    A selected-but-unusable live source falls back to fixtures with a warning
    rather than failing startup — a missing key should not take the app down.
    """
    if settings.marketplace_source != "serpapi":
        return None
    if settings.serpapi_key is None:
        _logger.warning(
            "marketplace.live_source_unavailable",
            extra={"reason": "serpapi_key_missing", "fallback": "fixtures"},
        )
        return None

    from .serpapi import SerpApiTransport

    _logger.info("marketplace.live_source_enabled", extra={"source": "serpapi"})
    return SerpApiTransport(
        settings.serpapi_key.get_secret_value(),
        timeout_seconds=settings.serpapi_timeout_seconds,
        country=settings.serpapi_country,
    )


def build_registry(
    *,
    cache: CacheProvider | None = None,
    settings: Settings | None = None,
) -> MarketplaceRegistry:
    resolved = get_settings() if settings is None else settings
    kwargs = {
        "cache": cache,
        "product_ttl_seconds": resolved.product_cache_ttl_seconds,
        "offer_ttl_seconds": resolved.offer_cache_ttl_seconds,
        "transport": build_transport(resolved),
    }
    return MarketplaceRegistry(
        {
            Marketplace.AMAZON: AmazonClient(**kwargs),
            Marketplace.FLIPKART: FlipkartClient(**kwargs),
        }
    )
