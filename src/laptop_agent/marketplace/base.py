"""Marketplace client contract.

The security property this contract exists to enforce: **a client owns its base
URL.** ``search`` accepts a validated
:class:`~laptop_agent.guardrails.tool_input.SearchProductsRequest`, which has no
field for a URL, a host, a path or headers. There is no code path by which model
output can cause a request to an arbitrary endpoint.

Clients return *raw* payloads. Validation is the caller's job, performed by
:class:`~laptop_agent.guardrails.tool_output.MarketplaceResponseValidator`, so
that a client cannot accidentally hand validated-looking data to the graph.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from ..cache.base import CacheProvider
from ..domain.enums import Marketplace
from ..guardrails.tool_input import FetchOffersRequest, SearchProductsRequest


@runtime_checkable
class MarketplaceClient(Protocol):
    """A read-only product/offer source."""

    @property
    def marketplace(self) -> Marketplace: ...

    @property
    def base_url(self) -> str:
        """Owned by the client. Never supplied by a caller or a model."""
        ...

    def search(self, request: SearchProductsRequest) -> dict[str, Any]:
        """Return a raw, unvalidated product payload."""
        ...

    def fetch_offers(self, request: FetchOffersRequest) -> dict[str, Any]:
        """Return a raw, unvalidated offer payload."""
        ...


class MarketplaceError(RuntimeError):
    """A provider failed. Carries the marketplace so one provider's outage can be
    reported without failing the whole search."""

    def __init__(self, marketplace: Marketplace, detail: str) -> None:
        self.marketplace = marketplace
        self.detail = detail
        super().__init__(f"{marketplace.value}: {detail}")


class CachingClientMixin:
    """Short-TTL caching of raw provider payloads.

    Prices are volatile, so the TTL is small and capped by the cache provider
    itself. The cache key includes every argument that affects the response —
    omitting one would serve a response for a different query.
    """

    marketplace: Marketplace

    def __init__(
        self,
        *,
        cache: CacheProvider | None = None,
        product_ttl_seconds: int = 300,
        offer_ttl_seconds: int = 120,
    ) -> None:
        self._cache = cache
        self._product_ttl = product_ttl_seconds
        self._offer_ttl = offer_ttl_seconds
        #: Per-client latency samples, surfaced in trace metadata.
        self.last_latency_ms: float = 0.0

    def _search_cache_key(self, request: SearchProductsRequest) -> str:
        budget = request.budget_max.amount if request.budget_max else "none"
        return (
            f"search:{self.marketplace.value}:{request.category.value}:"
            f"{request.query.lower()}:{request.max_results}:{request.currency.value}:{budget}"
        )

    def _offers_cache_key(self, request: FetchOffersRequest) -> str:
        ids = ",".join(sorted(request.product_ids))
        return f"offers:{self.marketplace.value}:{ids}"

    def _cached(self, key: str) -> dict[str, Any] | None:
        if self._cache is None:
            return None
        value = self._cache.get(key)
        return value if isinstance(value, dict) else None

    def _store(self, key: str, value: dict[str, Any], ttl: int) -> None:
        if self._cache is not None:
            self._cache.set(key, value, ttl)

    def _timed(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self.last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
