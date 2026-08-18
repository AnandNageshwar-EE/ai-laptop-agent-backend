"""In-memory TTL cache.

Deliberately the only implementation. Redis would add an operational dependency
for data whose maximum useful lifetime is five minutes; the spec says not to
introduce it until something actually needs it. When that day comes, implement
:class:`~laptop_agent.cache.base.CacheProvider` and inject it instead — no
calling code changes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from ..config import MAX_VOLATILE_CACHE_TTL_SECONDS
from .base import CacheStats


@dataclass(slots=True)
class _Entry:
    value: Any
    expires_at: float


class InMemoryCacheProvider:
    """Thread-safe TTL cache with a bounded size.

    ``ttl_seconds`` is clamped to the volatile-data ceiling, so no caller can
    accidentally pin a stale price by passing a large TTL.
    """

    def __init__(
        self,
        *,
        max_entries: int = 2_048,
        max_ttl_seconds: int = MAX_VOLATILE_CACHE_TTL_SECONDS,
        clock: Any = time.monotonic,
    ) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._max_ttl = max_ttl_seconds
        self._clock = clock
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats["misses"] += 1
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                return None
            self.stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        ttl = min(ttl_seconds, self._max_ttl)
        with self._lock:
            if len(self._entries) >= self._max_entries and key not in self._entries:
                self._evict_locked()
            self._entries[key] = _Entry(value=value, expires_at=self._clock() + ttl)
            self.stats["sets"] += 1

    def delete(self, key: str) -> None:
        with self._lock:
            if self._entries.pop(key, None) is not None:
                self.stats["evictions"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _evict_locked(self) -> None:
        """Drop expired entries; if none are expired, drop the soonest to expire."""
        now = self._clock()
        expired = [k for k, e in self._entries.items() if e.expires_at <= now]
        if expired:
            for key in expired:
                del self._entries[key]
                self.stats["expirations"] += 1
            return
        oldest = min(self._entries, key=lambda k: self._entries[k].expires_at)
        del self._entries[oldest]
        self.stats["evictions"] += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
