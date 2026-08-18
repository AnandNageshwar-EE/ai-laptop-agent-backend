"""Per-run metric collection.

Collected in-process and attached to the trace and the API response. Deliberately
small: counters and latencies, no payloads, so nothing here can leak content.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NodeTiming:
    node: str
    latency_ms: float
    error: str | None = None


@dataclass
class RunMetrics:
    """Accumulated facts about one graph run."""

    session_id: str
    node_timings: list[NodeTiming] = field(default_factory=list)
    llm_calls: int = 0
    llm_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    marketplaces_used: list[str] = field(default_factory=list)
    provider_cache_hits: int = 0
    provider_cache_misses: int = 0
    products_returned: int = 0
    products_quarantined: int = 0
    offers_quarantined: int = 0
    candidates_built: int = 0
    injection_flags: int = 0
    guardrail_blocks: int = 0
    validation_attempts: int = 0
    validation_failures: list[str] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)

    def record_node(self, node: str, latency_ms: float, error: str | None = None) -> None:
        self.node_timings.append(NodeTiming(node=node, latency_ms=latency_ms, error=error))

    def record_llm(self, stats: Any) -> None:
        """Fold in one :class:`~laptop_agent.llm.structured.InvocationStats`."""
        self.llm_calls += 1
        self.llm_retries += getattr(stats, "retries", 0)
        self.input_tokens += getattr(stats, "input_tokens", 0)
        self.output_tokens += getattr(stats, "output_tokens", 0)
        self.cache_read_tokens += getattr(stats, "cache_read_tokens", 0)
        self.cache_creation_tokens += getattr(stats, "cache_creation_tokens", 0)

    @property
    def total_latency_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 2)

    @property
    def prompt_cache_hit(self) -> bool:
        return self.cache_read_tokens > 0

    def as_trace_metadata(self) -> dict[str, Any]:
        """Flat, JSON-safe metadata for LangSmith. No content, no PII."""
        return {
            "total_latency_ms": self.total_latency_ms,
            "node_latencies_ms": {t.node: t.latency_ms for t in self.node_timings},
            "node_errors": {t.node: t.error for t in self.node_timings if t.error},
            "llm_calls": self.llm_calls,
            "retry_count": self.llm_retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_tokens,
            "cache_creation_input_tokens": self.cache_creation_tokens,
            "prompt_cache_hit": self.prompt_cache_hit,
            "marketplace_provider_count": len(self.marketplaces_used),
            "marketplaces_used": sorted(self.marketplaces_used),
            "product_cache_hits": self.provider_cache_hits,
            "product_cache_misses": self.provider_cache_misses,
            "products_returned": self.products_returned,
            "products_quarantined": self.products_quarantined,
            "offers_quarantined": self.offers_quarantined,
            "candidate_count": self.candidates_built,
            "injection_flags": self.injection_flags,
            "guardrail_blocks": self.guardrail_blocks,
            "recommendation_validation_attempts": self.validation_attempts,
            "recommendation_validation_failures": self.validation_failures,
        }
