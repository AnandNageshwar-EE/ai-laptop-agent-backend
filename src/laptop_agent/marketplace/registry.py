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
from .base import MarketplaceClient
from .clients import AmazonClient, FlipkartClient


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
    }
    return MarketplaceRegistry(
        {
            Marketplace.AMAZON: AmazonClient(**kwargs),
            Marketplace.FLIPKART: FlipkartClient(**kwargs),
        }
    )
