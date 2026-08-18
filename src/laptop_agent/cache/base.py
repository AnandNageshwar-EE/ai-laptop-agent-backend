"""Cache abstraction.

Two unrelated caching concerns exist in this application and they must not be
conflated:

**A. Prompt caching** — stable prompt prefixes. Handled by the provider
(Anthropic ``cache_control``) plus a local string cache for prompt *assembly*.
See :mod:`laptop_agent.prompts.provider`.

**B. Product / search / offer caching** — this module. Marketplace results are
volatile: prices and offers change. TTLs are therefore short and capped in
:class:`laptop_agent.config.Settings`, and price data is always re-fetched
rather than served from a long-lived store.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CacheProvider(Protocol):
    """Minimal cache surface. Implementations must be safe to share."""

    def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` if absent or expired."""
        ...

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store ``value`` under ``key`` for at most ``ttl_seconds``."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        ...


class CacheStats(dict[str, int]):
    """Hit/miss counters, surfaced in trace metadata for cache-effectiveness checks."""

    def __init__(self) -> None:
        super().__init__(hits=0, misses=0, sets=0, evictions=0, expirations=0)

    @property
    def hit_rate(self) -> float:
        total = self["hits"] + self["misses"]
        return self["hits"] / total if total else 0.0
