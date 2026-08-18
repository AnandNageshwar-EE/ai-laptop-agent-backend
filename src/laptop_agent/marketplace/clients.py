"""Amazon and Flipkart clients.

Both are simulated against :mod:`.fixtures`. The structure is the shape a real
client would take, and the important part is already real: the base URL is a
class attribute, the only input is a validated request object, and the returned
payload is raw.

To go live, replace ``_fetch_products`` / ``_fetch_offers`` with an HTTP call
built from ``self.base_url`` plus a fixed path. Nothing else changes, because
every consumer already treats the response as untrusted.
"""

from __future__ import annotations

from typing import Any

from ..cache.base import CacheProvider
from ..domain.enums import Marketplace
from ..guardrails.tool_input import FetchOffersRequest, SearchProductsRequest
from .base import CachingClientMixin, ProductTransport
from .fixtures import (
    AMAZON_OFFERS,
    AMAZON_PRODUCTS,
    FLIPKART_OFFERS,
    FLIPKART_PRODUCTS,
    filter_products,
)


class _FixtureBackedClient(CachingClientMixin):
    """Shared implementation for the simulated clients."""

    _marketplace: Marketplace
    _base_url: str
    _products: list[dict[str, Any]]
    _offers: list[dict[str, Any]]

    def __init__(
        self,
        *,
        cache: CacheProvider | None = None,
        product_ttl_seconds: int = 300,
        offer_ttl_seconds: int = 120,
        transport: ProductTransport | None = None,
    ) -> None:
        self.marketplace = self._marketplace
        #: When set, listings come from a live source instead of the fixtures.
        #: Offers still come from fixtures — see ``_fetch_offers``.
        self._transport = transport
        super().__init__(
            cache=cache,
            product_ttl_seconds=product_ttl_seconds,
            offer_ttl_seconds=offer_ttl_seconds,
        )

    @property
    def base_url(self) -> str:
        """Fixed per client. Not configurable by a caller or a model."""
        return self._base_url

    # ----- public API -----

    def search(self, request: SearchProductsRequest) -> dict[str, Any]:
        if request.marketplace is not self._marketplace:
            raise ValueError(
                f"{type(self).__name__} received a request for {request.marketplace}"
            )
        key = self._search_cache_key(request)
        cached = self._cached(key)
        if cached is not None:
            return {**cached, "cache_hit": True}

        payload = self._timed(self._fetch_products, request)
        self._store(key, payload, self._product_ttl)
        return {**payload, "cache_hit": False}

    def fetch_offers(self, request: FetchOffersRequest) -> dict[str, Any]:
        if request.marketplace is not self._marketplace:
            raise ValueError(
                f"{type(self).__name__} received a request for {request.marketplace}"
            )
        key = self._offers_cache_key(request)
        cached = self._cached(key)
        if cached is not None:
            return {**cached, "cache_hit": True}

        payload = self._timed(self._fetch_offers, request)
        # Offers move faster than listings, hence the shorter TTL.
        self._store(key, payload, self._offer_ttl)
        return {**payload, "cache_hit": False}

    # ----- replace these two methods with real HTTP calls -----

    def _fetch_products(self, request: SearchProductsRequest) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport.fetch_products(self._marketplace, request)
        matched = filter_products(
            self._products, query=request.query, max_results=request.max_results
        )
        return {
            "marketplace": self._marketplace.value,
            "source": f"{self._base_url}/search",
            "products": matched,
        }

    def _fetch_offers(self, request: FetchOffersRequest) -> dict[str, Any]:
        """Offers.

        Live search APIs do not expose bank/exchange/cashback offer structures,
        so with a live transport there are simply no offers rather than invented
        ones. The effective price then equals the real listed price, which is
        correct: a discount we cannot verify must never be applied.
        """
        if self._transport is not None:
            return {
                "marketplace": self._marketplace.value,
                "source": "live",
                "offers": [],
            }
        wanted = set(request.product_ids)
        return {
            "marketplace": self._marketplace.value,
            "source": f"{self._base_url}/offers",
            "offers": [
                offer
                for offer in self._offers
                # The orphan-offer fixture is returned deliberately so the
                # "offer for a product we never retrieved" path is exercised.
                if offer.get("product_id") in wanted
                or offer.get("offer_id", "").endswith("ORPHAN")
            ],
        }


class AmazonClient(_FixtureBackedClient):
    _marketplace = Marketplace.AMAZON
    _base_url = "https://www.amazon.in/api"
    _products = AMAZON_PRODUCTS
    _offers = AMAZON_OFFERS


class FlipkartClient(_FixtureBackedClient):
    _marketplace = Marketplace.FLIPKART
    _base_url = "https://www.flipkart.com/api"
    _products = FLIPKART_PRODUCTS
    _offers = FLIPKART_OFFERS
