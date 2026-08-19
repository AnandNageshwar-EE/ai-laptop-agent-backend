"""SerpApi response mapping.

Uses recorded payloads in the shape SerpApi actually returns, so the mapping is
tested without network access. What matters here is that live third-party data
lands in the *same* untrusted envelope the fixtures use, and is then subject to
the identical validation — including the host allowlist that rejects a link
pointing anywhere but the claimed marketplace.
"""

from __future__ import annotations

import pytest

from laptop_agent.domain.enums import Marketplace
from laptop_agent.guardrails.tool_input import SearchProductsRequest
from laptop_agent.guardrails.tool_output import MarketplaceResponseValidator
from laptop_agent.marketplace.serpapi import SerpApiTransport
from laptop_agent.marketplace.spec_parser import parse_specs

AMAZON_PAYLOAD = {
    "organic_results": [
        {
            "asin": "B0CX5X9V4T",
            "title": 'Lenovo IdeaPad Slim 3 AMD Ryzen 5 7520U 15.6" (16GB/512GB SSD/Windows 11 Home) Laptop',
            "link": "https://www.amazon.in/dp/B0CX5X9V4T",
            "price": "₹42,990",
            "extracted_price": 42990,
            "old_price": "₹66,890",
            "extracted_old_price": 66890,
            "rating": 4.2,
            "reviews": 1834,
        },
        {
            "asin": "B0CHX2F5QT",
            "title": "HP Victus Gaming Laptop, Intel Core i5-12450H, RTX 3050 (16GB DDR4/512GB SSD/144Hz/Win11)",
            # A tracking redirect rather than a product URL.
            "link": "https://www.amazon.in/sspa/click?ie=UTF8&spc=abc&url=%2Fdp%2FB0CHX2F5QT",
            "extracted_price": 61990,
            "rating": 4.1,
            "reviews": 902,
        },
        {
            # Accessory, not a laptop.
            "asin": "B08XYZACCS",
            "title": "Laptop Sleeve Case 15.6 inch, Water Resistant",
            "link": "https://www.amazon.in/dp/B08XYZACCS",
            "extracted_price": 799,
        },
        {
            # No price — unusable.
            "asin": "B0NOPRICE1",
            "title": "Some Laptop 16GB 512GB SSD",
            "link": "https://www.amazon.in/dp/B0NOPRICE1",
        },
        {
            # No ASIN — no stable identifier.
            "title": "Unidentified Laptop 8GB 256GB SSD",
            "link": "https://www.amazon.in/dp/UNKNOWN",
            "extracted_price": 29990,
        },
    ]
}

#: Organic Google results restricted to flipkart.com. This is the strategy that
#: works: google_shopping returns Flipkart prices but only a Google catalog link,
#: which no provenance check can verify.
FLIPKART_ORGANIC_PAYLOAD = {
    "organic_results": [
        {
            "title": "ASUS Vivobook 16 Ryzen 7 7730U 16 GB 512 GB SSD Windows 11 - Flipkart.com",
            "link": "https://www.flipkart.com/asus-vivobook-16-amd-ryzen-7-7730u-16-gb-512-gb-ssd-windows-11/p/itm9f8a1b2c3d4e5?pid=COMGZ8KHFYRGH2XM",
            "snippet": "ASUS Vivobook 16 ... ₹54,990 ... 4.3 out of 5 stars",
        },
        {
            # A category listing page, not a product.
            "title": "16GB RAM Laptops - Buy 16GB RAM Laptops Online",
            "link": "https://www.flipkart.com/laptops/~16gb-ram-laptops/pr?sid=6bo,b5g",
            "snippet": "Shop laptops from ₹31,990",
        },
        {
            # A product page with no price anywhere — cannot be compared.
            "title": "HP 15s Intel Core i5 16 GB 512 GB SSD",
            "link": "https://www.flipkart.com/hp-15s-intel-core-i5-16-gb-512-gb-ssd/p/itmabc123def456",
            "snippet": "Specifications and reviews",
        },
        {
            # Wrong host entirely.
            "title": "Dell Inspiron 15 16GB 512GB SSD",
            "link": "https://www.croma.com/dell-inspiron-15/p/123456",
            "snippet": "₹58,990",
        },
    ]
}


@pytest.fixture
def transport() -> SerpApiTransport:
    return SerpApiTransport("test-key-not-used")


@pytest.fixture
def request_obj() -> SearchProductsRequest:
    return SearchProductsRequest(
        query="laptop 16GB SSD", marketplace=Marketplace.AMAZON, max_results=10
    )


# --- amazon mapping -------------------------------------------------------

def test_amazon_mapping_keeps_only_usable_laptops(transport, request_obj):
    products = transport._map_amazon(AMAZON_PAYLOAD, limit=10)
    ids = [p["product_id"] for p in products]
    # Two usable laptops; accessory, priceless and id-less rows are dropped.
    assert ids == ["AMZ-B0CX5X9V4T", "AMZ-B0CHX2F5QT"]


def test_amazon_prices_come_from_the_numeric_field(transport):
    products = transport._map_amazon(AMAZON_PAYLOAD, limit=10)
    first = products[0]
    assert first["price"] == {"amount": "42990.00", "currency": "INR"}
    assert first["mrp"] == {"amount": "66890.00", "currency": "INR"}


def test_amazon_urls_are_canonicalised_from_the_asin(transport):
    """Live links carry the search session and ad redirects; neither is a product URL."""
    products = transport._map_amazon(AMAZON_PAYLOAD, limit=10)
    for product in products:
        asin = product["product_id"].removeprefix("AMZ-")
        assert product["url"] == f"https://www.amazon.in/dp/{asin}"
        assert "sspa/click" not in product["url"]
        assert "/ref=" not in product["url"]
        assert "dib=" not in product["url"]


def test_specs_are_parsed_from_the_title_not_invented(transport):
    products = transport._map_amazon(AMAZON_PAYLOAD, limit=10)
    specs = products[0]["specs"]
    assert specs["ram_gb"] == 16
    assert specs["storage_gb"] == 512
    assert specs["storage_type"] == "ssd"
    # Weight and battery are not in the title, so they must be absent entirely.
    assert "weight_kg" not in specs
    assert "battery_hours" not in specs


def test_limit_is_respected(transport):
    assert len(transport._map_amazon(AMAZON_PAYLOAD, limit=1)) == 1


# --- flipkart via google shopping ----------------------------------------

def test_flipkart_mapping_keeps_only_priced_product_pages(transport):
    products = transport._map_flipkart_organic(FLIPKART_ORGANIC_PAYLOAD, limit=10)
    assert len(products) == 1
    product = products[0]
    assert "flipkart.com" in product["url"]
    assert "/p/" in product["url"]
    assert product["price"] == {"amount": "54990.00", "currency": "INR"}


def test_flipkart_category_pages_and_other_merchants_are_dropped(transport):
    products = transport._map_flipkart_organic(FLIPKART_ORGANIC_PAYLOAD, limit=10)
    urls = " ".join(p["url"] for p in products)
    assert "/pr?" not in urls
    assert "croma.com" not in urls


def test_flipkart_row_without_a_price_is_dropped(transport):
    """A listing whose price cannot be established is useless to this agent."""
    products = transport._map_flipkart_organic(FLIPKART_ORGANIC_PAYLOAD, limit=10)
    assert all("hp-15s" not in p["url"] for p in products)


def test_flipkart_specs_come_from_the_url_slug_and_snippet(transport):
    products = transport._map_flipkart_organic(FLIPKART_ORGANIC_PAYLOAD, limit=10)
    specs = products[0]["specs"]
    assert specs["ram_gb"] == 16
    assert specs["storage_gb"] == 512


def test_marketplace_suffix_is_stripped_from_the_title(transport):
    products = transport._map_flipkart_organic(FLIPKART_ORGANIC_PAYLOAD, limit=10)
    assert "Flipkart.com" not in products[0]["title"]


# --- the whole point: live data goes through the same validation ----------

def test_mapped_amazon_payload_passes_domain_validation(transport):
    payload = {"products": transport._map_amazon(AMAZON_PAYLOAD, limit=10)}
    outcome = MarketplaceResponseValidator(Marketplace.AMAZON).validate_products(payload)
    assert outcome.accepted_count == 2
    assert outcome.quarantined_count == 0
    for product in outcome.accepted:
        assert str(product.url).startswith("https://www.amazon.in/dp/")


def test_mapped_flipkart_payload_passes_domain_validation(transport):
    payload = {"products": transport._map_flipkart_organic(FLIPKART_ORGANIC_PAYLOAD, limit=10)}
    outcome = MarketplaceResponseValidator(Marketplace.FLIPKART).validate_products(payload)
    assert outcome.accepted_count == 1


def test_an_offsite_link_that_slips_through_is_still_quarantined(transport):
    """Defence in depth: the transport filters, the validator enforces."""
    forged = transport._map_amazon(AMAZON_PAYLOAD, limit=1)
    forged[0]["url"] = "https://phishing.example.com/dp/B0CX5X9V4T"
    outcome = MarketplaceResponseValidator(Marketplace.AMAZON).validate_products(
        {"products": forged}
    )
    assert outcome.accepted_count == 0
    assert "untrusted_url_host" in outcome.quarantined[0][1]


# --- failure handling -----------------------------------------------------

def test_unexpected_response_shapes_yield_no_products(transport):
    for payload in ({}, {"organic_results": "not a list"}, {"organic_results": [1, 2]}):
        assert transport._map_amazon(payload, limit=10) == []


def test_network_failure_returns_an_empty_envelope(monkeypatch, transport, request_obj):
    def boom(_params):
        raise TimeoutError("connection timed out")

    monkeypatch.setattr(transport, "_call", boom)
    result = transport.fetch_products(Marketplace.AMAZON, request_obj)
    # One provider failing must never fail the search.
    assert result["products"] == []
    assert result["marketplace"] == "amazon"


def test_api_error_does_not_echo_the_request(monkeypatch, transport, request_obj):
    """SerpApi echoes the request in errors, and the request contains the key."""
    import json as json_module

    class FakeResponse:
        def read(self):
            return json_module.dumps(
                {"error": "Invalid API key: test-key-not-used"}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: FakeResponse()
    )
    with pytest.raises(RuntimeError) as excinfo:
        transport._call({"engine": "amazon"})
    assert "test-key-not-used" not in str(excinfo.value)


# --- configuration --------------------------------------------------------

def test_missing_key_falls_back_to_fixtures_rather_than_failing():
    from laptop_agent.config import Settings
    from laptop_agent.marketplace.registry import build_transport

    settings = Settings(marketplace_source="serpapi", serpapi_key=None)
    assert build_transport(settings) is None
    assert settings.effective_marketplace_source == "fixtures"
    assert not settings.live_marketplace_enabled


def test_live_source_is_reported_in_trace_metadata():
    from laptop_agent.config import Settings

    settings = Settings(marketplace_source="serpapi", serpapi_key="k" * 20)
    assert settings.live_marketplace_enabled
    assert settings.base_trace_metadata()["marketplace_source"] == "serpapi"


def test_live_transport_reports_no_offers_rather_than_inventing_them():
    from laptop_agent.guardrails.tool_input import FetchOffersRequest
    from laptop_agent.marketplace.clients import AmazonClient

    client = AmazonClient(transport=SerpApiTransport("key"))
    result = client.fetch_offers(
        FetchOffersRequest(marketplace=Marketplace.AMAZON, product_ids=["AMZ-B0CX5X9V4T"])
    )
    # A discount that cannot be verified must never be applied.
    assert result["offers"] == []


def test_spec_parser_never_invents_a_missing_value():
    specs = parse_specs("Some Laptop With No Stated Specifications")
    assert specs.ram_gb is None
    assert specs.storage_gb is None
    assert specs.weight_kg is None
    assert not specs.is_sufficient


# ---------------------------------------------------------------------------
# Hermeticity guard.
#
# The suite once started making live SerpApi calls because a developer's local
# .env set MARKETPLACE_SOURCE=serpapi. That made it slow, non-deterministic and
# quietly expensive. These tests fail loudly if that isolation ever regresses.
# ---------------------------------------------------------------------------


def test_suite_runs_against_fixtures_never_a_live_api():
    from laptop_agent.config import get_settings

    settings = get_settings()
    assert settings.effective_marketplace_source == "fixtures"
    assert not settings.live_marketplace_enabled


def test_registry_built_in_tests_has_no_live_transport():
    from laptop_agent.config import get_settings
    from laptop_agent.marketplace.registry import build_transport

    assert build_transport(get_settings()) is None


def test_no_outbound_http_during_a_full_agent_run(monkeypatch):
    """Any attempt to open a socket during a run is a test-isolation failure."""
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("test attempted a live HTTP request")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    from laptop_agent.agent import LaptopAgent

    reply = LaptopAgent().run(
        message="laptop for software development under 80000 with 16GB RAM"
    )
    assert reply.recommendation is not None


# ---------------------------------------------------------------------------
# Provider routing. OpenRouter is OpenAI-shaped, so the usage fields and the
# caching guarantees differ from the native Anthropic path.
# ---------------------------------------------------------------------------


def test_openrouter_reports_caching_only_for_models_that_can_cache():
    from laptop_agent.config import Settings

    cacheable = Settings(
        llm_mode="live", llm_provider="openrouter",
        openrouter_model="anthropic/claude-opus-5", openrouter_api_key="k" * 20,
    )
    assert cacheable.provider_prompt_caching_supported

    free = Settings(
        llm_mode="live", llm_provider="openrouter",
        openrouter_model="meta-llama/llama-3.3-70b-instruct:free",
        openrouter_api_key="k" * 20,
    )
    # A free open-weight model has no server-side prompt cache. Reporting
    # otherwise would imply token savings that are not happening.
    assert not free.provider_prompt_caching_supported


def test_offline_mode_never_claims_provider_caching():
    from laptop_agent.config import Settings

    assert not Settings(llm_mode="offline").provider_prompt_caching_supported


def test_traced_model_name_reflects_the_openrouter_slug():
    from laptop_agent.config import Settings

    settings = Settings(
        llm_mode="live", llm_provider="openrouter",
        openrouter_model="anthropic/claude-opus-5", openrouter_api_key="k" * 20,
    )
    assert settings.traced_model_name == "anthropic/claude-opus-5"
    assert settings.base_trace_metadata()["provider_prompt_caching"] == "true"


def test_openrouter_requires_a_key():
    from laptop_agent.config import Settings
    from laptop_agent.llm.provider import LLMUnavailableError, build_chat_model

    with pytest.raises(LLMUnavailableError):
        build_chat_model(
            Settings(llm_mode="live", llm_provider="openrouter", openrouter_api_key=None)
        )


def test_openrouter_usage_fields_are_parsed():
    """OpenRouter names cache counters differently from Anthropic.

    Without handling prompt_tokens_details.cached_tokens the cache would appear
    to never hit, and the whole point of the stable prefix would be unverifiable.
    """
    from laptop_agent.llm.structured import InvocationStats, StructuredLLM

    class FakeResponse:
        usage_metadata: dict = {}
        response_metadata = {
            "token_usage": {
                "prompt_tokens": 1800,
                "completion_tokens": 210,
                "prompt_tokens_details": {"cached_tokens": 1665},
                "cache_write_tokens": 0,
            }
        }

    stats = InvocationStats(schema="RequirementExtraction")
    StructuredLLM._record_usage(stats, FakeResponse())
    assert stats.cache_read_tokens == 1665
    assert stats.input_tokens == 1800
    assert stats.output_tokens == 210
    assert stats.cache_hit


def test_anthropic_native_usage_fields_still_parsed():
    from laptop_agent.llm.structured import InvocationStats, StructuredLLM

    class FakeResponse:
        usage_metadata = {
            "input_tokens": 1700,
            "output_tokens": 200,
            "input_token_details": {"cache_read": 1650, "cache_creation": 0},
        }
        response_metadata: dict = {}

    stats = InvocationStats(schema="RequirementExtraction")
    StructuredLLM._record_usage(stats, FakeResponse())
    assert stats.cache_read_tokens == 1650
    assert stats.input_tokens == 1700


def test_openrouter_key_shape_is_redacted():
    from laptop_agent.security.redaction import SecretRedactor

    key = "sk-or-v1-" + "a1b2c3d4" * 8
    assert key not in SecretRedactor().redact_text(f"OPENROUTER_API_KEY={key}")


def test_gateway_providers_try_tool_calling_first():
    """Routed through OpenRouter, response_format is not strictly enforced.

    Observed with anthropic/claude-opus-5 via OpenRouter: the json_schema
    attempt returned an invented shape (budget:extra_forbidden,
    trade_offs.0:model_type) and only the tool-calling retry succeeded. Putting
    tool-calling first removes a guaranteed wasted round trip.
    """
    from laptop_agent.llm.structured import strategies_for

    assert strategies_for("openrouter")[0] == "function_calling"
    assert strategies_for("anthropic")[0] == "json_schema"


def test_runner_up_note_cap_matches_what_the_node_sends():
    """A schema cap below the payload size rejects a correct response."""
    from laptop_agent.domain.recommendation import Recommendation
    from laptop_agent.llm.schemas import RecommendationExplanation

    notes = RecommendationExplanation.model_fields["runner_up_notes"].metadata
    runners = Recommendation.model_fields["runner_ups"].metadata
    cap = next(m.max_length for m in notes if hasattr(m, "max_length"))
    sent = next(m.max_length for m in runners if hasattr(m, "max_length"))
    assert cap >= sent, f"schema caps notes at {cap} but up to {sent} runner-ups are sent"


def test_bare_string_trade_offs_are_normalised_not_rejected():
    """Models return trade_offs as sentences about one call in eight.

    Rejecting a usable explanation over the container shape sent the request to
    the deterministic explainer instead. Normalising keeps the bounds intact.
    """
    from laptop_agent.llm.schemas import RecommendationExplanation

    parsed = RecommendationExplanation.model_validate(
        {
            "rationale": "This laptop fits the stated requirements well.",
            "trade_offs": [
                "Heavier than the alternatives at over 2kg.",
                {"dimension": "battery life", "detail": "Shorter rated battery life."},
                "",
            ],
        }
    )
    assert len(parsed.trade_offs) == 2
    assert parsed.trade_offs[0].dimension == "trade-off"
    assert "Heavier" in parsed.trade_offs[0].detail
    assert parsed.trade_offs[1].dimension == "battery life"


def test_normalised_trade_off_still_respects_length_bounds():
    from laptop_agent.llm.schemas import RecommendationExplanation

    parsed = RecommendationExplanation.model_validate(
        {"rationale": "Valid rationale text here.", "trade_offs": ["x" * 5000]}
    )
    assert len(parsed.trade_offs[0].detail) <= 280
