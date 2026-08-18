"""Both caching concerns: volatile product data, and stable prompt prefixes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from laptop_agent.cache import InMemoryCacheProvider
from laptop_agent.config import MAX_VOLATILE_CACHE_TTL_SECONDS, Settings
from laptop_agent.prompts.provider import CachedPromptProvider, PromptTask


# --- product/search cache (volatile) --------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_get_set_delete():
    cache = InMemoryCacheProvider()
    assert cache.get("missing") is None
    cache.set("k", {"v": 1}, 60)
    assert cache.get("k") == {"v": 1}
    cache.delete("k")
    assert cache.get("k") is None


def test_entries_expire():
    clock = FakeClock()
    cache = InMemoryCacheProvider(clock=clock)
    cache.set("price", {"amount": 1}, 60)
    clock.advance(59)
    assert cache.get("price") is not None
    clock.advance(2)
    assert cache.get("price") is None


def test_ttl_is_clamped_to_the_volatile_ceiling():
    """A caller cannot pin a stale price by asking for a long TTL."""
    clock = FakeClock()
    cache = InMemoryCacheProvider(clock=clock, max_ttl_seconds=300)
    cache.set("price", {"amount": 1}, 86_400)
    clock.advance(301)
    assert cache.get("price") is None


def test_settings_refuse_long_lived_price_caches():
    with pytest.raises(ValidationError):
        Settings(product_cache_ttl_seconds=MAX_VOLATILE_CACHE_TTL_SECONDS + 1)
    with pytest.raises(ValidationError):
        Settings(offer_cache_ttl_seconds=3600)
    # And the default is inside the ceiling.
    assert Settings().product_cache_ttl_seconds <= MAX_VOLATILE_CACHE_TTL_SECONDS


def test_size_is_bounded():
    cache = InMemoryCacheProvider(max_entries=10)
    for index in range(50):
        cache.set(f"k{index}", index, 60)
    assert len(cache) <= 10


def test_stats_track_hits_and_misses():
    cache = InMemoryCacheProvider()
    cache.set("k", 1, 60)
    cache.get("k")
    cache.get("nope")
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1
    assert cache.stats.hit_rate == 0.5


def test_marketplace_client_uses_the_cache(registry, cache):
    from laptop_agent.domain.enums import Marketplace
    from laptop_agent.guardrails.tool_input import SearchProductsRequest

    client = registry.get(Marketplace.AMAZON)
    request = SearchProductsRequest(query="laptop 16gb", marketplace=Marketplace.AMAZON)
    first = client.search(request)
    second = client.search(request)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_different_queries_do_not_share_a_cache_entry(registry):
    from laptop_agent.domain.enums import Marketplace
    from laptop_agent.guardrails.tool_input import SearchProductsRequest

    client = registry.get(Marketplace.AMAZON)
    client.search(SearchProductsRequest(query="gaming laptop", marketplace=Marketplace.AMAZON))
    other = client.search(
        SearchProductsRequest(query="lightweight laptop", marketplace=Marketplace.AMAZON)
    )
    assert other["cache_hit"] is False


# --- prompt caching (stable prefixes) ------------------------------------

@pytest.fixture
def prompts() -> CachedPromptProvider:
    return CachedPromptProvider(caching_enabled=True, cache_ttl="5m")


def test_two_cache_breakpoints_are_placed(prompts):
    blocks = prompts.system_blocks(PromptTask.REQUIREMENTS)
    assert len(blocks) == 2
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in blocks)


def test_shared_prefix_is_identical_across_tasks(prompts):
    """This is what makes the big prefix cacheable once and read by every task."""
    prefixes = {
        prompts.system_blocks(task)[0]["text"] for task in PromptTask
    }
    assert len(prefixes) == 1


def test_prefix_is_byte_stable_across_calls(prompts):
    first = prompts.system_blocks(PromptTask.REQUIREMENTS)[0]["text"]
    second = prompts.system_blocks(PromptTask.RECOMMENDATION)[0]["text"]
    third = prompts.system_blocks(PromptTask.REQUIREMENTS)[0]["text"]
    assert first == second == third
    assert prompts.prefix_fingerprint() == prompts.prefix_fingerprint()


def test_blocks_are_fresh_objects_so_callers_cannot_corrupt_them(prompts):
    first = prompts.system_blocks(PromptTask.REQUIREMENTS)
    first[0]["text"] = "MUTATED"
    second = prompts.system_blocks(PromptTask.REQUIREMENTS)
    assert second[0]["text"] != "MUTATED"


def test_prefix_exceeds_the_minimum_cacheable_size(prompts):
    """Below the model's minimum, a marked prefix silently never caches."""
    prefix = prompts.stable_prefix()
    # ~4 chars per token is a conservative estimate; the Opus 5 minimum is 512.
    assert len(prefix) / 4 > 512


def test_prefix_contains_no_request_specific_content(prompts):
    prefix = prompts.stable_prefix()
    for volatile in (
        "sess_", "80000", "₹", "hdfc", "amazon.in/dp", "2026", "budget_max=",
    ):
        assert volatile.lower() not in prefix.lower(), f"volatile content in prefix: {volatile}"


def test_caching_can_be_disabled():
    provider = CachedPromptProvider(caching_enabled=False)
    blocks = provider.system_blocks(PromptTask.REQUIREMENTS)
    assert all("cache_control" not in block for block in blocks)


def test_one_hour_ttl_is_expressed_in_the_breakpoint():
    provider = CachedPromptProvider(caching_enabled=True, cache_ttl="1h")
    blocks = provider.system_blocks(PromptTask.REQUIREMENTS)
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_unsupported_ttl_is_rejected():
    with pytest.raises(ValueError):
        CachedPromptProvider(cache_ttl="7d")


def test_prompt_versions_are_reported_for_tracing(prompts):
    versions = prompts.versions(PromptTask.RECOMMENDATION)
    assert versions["base_prompt_version"] == "v1"
    assert versions["task_prompt_version"] == "v1"
    assert versions["prompt_task"] == "recommendation"
    assert len(versions["prompt_prefix_fingerprint"]) == 16


def test_assembly_cache_is_used(prompts):
    prompts.stable_prefix()
    before = prompts.assembly_hits
    prompts.stable_prefix()
    prompts.stable_prefix()
    assert prompts.assembly_hits > before
