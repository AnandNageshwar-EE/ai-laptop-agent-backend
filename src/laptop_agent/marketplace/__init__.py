"""Marketplace providers. Clients own their base URLs; responses are untrusted."""

from .base import CachingClientMixin, MarketplaceClient, MarketplaceError
from .clients import AmazonClient, FlipkartClient
from .registry import MarketplaceRegistry, build_registry

__all__ = [
    "AmazonClient",
    "CachingClientMixin",
    "FlipkartClient",
    "MarketplaceClient",
    "MarketplaceError",
    "MarketplaceRegistry",
    "build_registry",
]
