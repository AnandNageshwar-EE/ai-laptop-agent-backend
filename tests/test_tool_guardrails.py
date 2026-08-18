"""Tool argument and tool response guardrails."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from laptop_agent.domain.enums import Currency, Marketplace
from laptop_agent.domain.money import Money
from laptop_agent.guardrails.tool_input import (
    MAX_QUERY_CHARS,
    MAX_RESULTS_HARD_LIMIT,
    FetchOffersRequest,
    SearchProductsRequest,
)
from laptop_agent.guardrails.tool_output import (
    TRUSTED_HOSTS,
    MarketplaceResponseValidator,
)


# --- tool input ------------------------------------------------------------

def test_valid_search_request():
    request = SearchProductsRequest(
        query="laptop 16GB SSD", marketplace=Marketplace.AMAZON, max_results=5
    )
    assert request.query == "laptop 16GB SSD"
    assert request.max_results == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "a"},                                   # too short
        {"query": "x" * (MAX_QUERY_CHARS + 1)},           # too long
        {"query": "laptop", "max_results": 0},            # below range
        {"query": "laptop", "max_results": MAX_RESULTS_HARD_LIMIT + 1},
        {"query": "laptop", "marketplace": "ebay"},       # not an allowed value
        {"query": "!!!"},                                 # nothing searchable
    ],
)
def test_rejects_invalid_search_arguments(kwargs):
    payload = {"marketplace": Marketplace.AMAZON, **kwargs}
    with pytest.raises(ValidationError):
        SearchProductsRequest(**payload)


def test_search_request_has_no_url_field():
    """The model must not be able to steer a request at an arbitrary endpoint."""
    fields = set(SearchProductsRequest.model_fields)
    for forbidden in ("url", "base_url", "endpoint", "host", "path", "headers", "body"):
        assert forbidden not in fields


def test_extra_arguments_are_rejected_not_ignored():
    with pytest.raises(ValidationError):
        SearchProductsRequest(
            query="laptop",
            marketplace=Marketplace.AMAZON,
            url="https://evil.example.com",
        )


def test_query_is_sanitised_of_syntax_characters():
    request = SearchProductsRequest(
        query="laptop; DROP TABLE products; <script>x</script>",
        marketplace=Marketplace.AMAZON,
    )
    for char in (";", "<", ">"):
        assert char not in request.query


def test_currency_must_agree_with_budget():
    with pytest.raises(ValidationError):
        SearchProductsRequest(
            query="laptop",
            marketplace=Marketplace.AMAZON,
            budget_max=Money(amount=1000, currency=Currency.USD),
            currency=Currency.INR,
        )


def test_offers_request_requires_wellformed_ids():
    with pytest.raises(ValidationError):
        FetchOffersRequest(marketplace=Marketplace.AMAZON, product_ids=["../../etc/passwd"])
    request = FetchOffersRequest(
        marketplace=Marketplace.AMAZON, product_ids=["AMZ-1234", "AMZ-1234"]
    )
    assert request.product_ids == ["AMZ-1234"]


# --- tool output -----------------------------------------------------------

def _base_product(**overrides) -> dict:
    product = {
        "product_id": "AMZ-TEST-1",
        "title": "Test Laptop 14",
        "brand": "TestBrand",
        "url": "https://www.amazon.in/dp/TEST1",
        "price": {"amount": "50000.00", "currency": "INR"},
        "in_stock": True,
        "specs": {
            "ram_gb": 16, "storage_gb": 512, "storage_type": "ssd",
            "cpu": "Test CPU", "screen_inches": 14.0, "weight_kg": 1.4,
            "battery_hours": 8.0, "os": "windows",
        },
    }
    product.update(overrides)
    return product


@pytest.fixture
def validator() -> MarketplaceResponseValidator:
    return MarketplaceResponseValidator(Marketplace.AMAZON)


def test_accepts_wellformed_product(validator):
    outcome = validator.validate_products({"products": [_base_product()]})
    assert outcome.accepted_count == 1


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"product_id": None}, "missing_product_id"),
        ({"product_id": ""}, "missing_product_id"),
        ({"title": None}, "missing_title"),
        ({"price": None}, "missing_price"),
        ({"price": {"amount": "-100", "currency": "INR"}}, "invalid_price"),
        ({"price": {"amount": "0", "currency": "INR"}}, "non_positive_price"),
        # An unsupported currency fails during payload parsing, one layer earlier
        # than the price check — which is the desired order.
        ({"price": {"amount": "100", "currency": "XYZ"}}, "malformed_payload"),
        ({"url": None}, "missing_url"),
        ({"url": "http://www.amazon.in/dp/X"}, "disallowed_url_scheme"),
        ({"url": "https://evil.example.com/dp/X"}, "untrusted_url_host"),
        ({"url": "javascript:alert(1)"}, "disallowed_url_scheme"),
        ({"specs": {}}, "missing_specs"),
        (
            {"price": {"amount": "60000", "currency": "INR"},
             "mrp": {"amount": "50000", "currency": "INR"}},
            "listed_price_exceeds_mrp",
        ),
        (
            {"price": {"amount": "29990", "currency": "INR"},
             "mrp": {"amount": "1299990", "currency": "INR"}},
            "implausible_mrp_discount",
        ),
        (
            {"price": {"amount": "1000", "currency": "USD"},
             "mrp": {"amount": "1200", "currency": "INR"}},
            "currency_mismatch_price_vs_mrp",
        ),
    ],
)
def test_quarantines_malformed_products(validator, overrides, expected):
    outcome = validator.validate_products({"products": [_base_product(**overrides)]})
    assert outcome.accepted_count == 0
    assert outcome.quarantined_count == 1
    assert expected in outcome.quarantined[0][1]


def test_rejects_unexpected_envelope(validator):
    for payload in ("not a list", {"unexpected": "shape"}, 42, None):
        outcome = validator.validate_products(payload)
        assert outcome.accepted_count == 0
        assert outcome.quarantined[0][1] == "unexpected_response_structure"


def test_duplicate_product_ids_are_quarantined(validator):
    outcome = validator.validate_products(
        {"products": [_base_product(), _base_product()]}
    )
    assert outcome.accepted_count == 1
    assert "duplicate_product_id" in outcome.quarantined[0][1]


def test_trusted_hosts_are_per_marketplace(validator):
    # An Amazon URL must not validate for Flipkart, and vice versa.
    flipkart = MarketplaceResponseValidator(Marketplace.FLIPKART)
    outcome = flipkart.validate_products({"products": [_base_product()]})
    assert outcome.accepted_count == 0
    assert "untrusted_url_host" in outcome.quarantined[0][1]
    assert TRUSTED_HOSTS[Marketplace.AMAZON].isdisjoint(TRUSTED_HOSTS[Marketplace.FLIPKART])


def test_offer_for_unknown_product_is_quarantined(validator):
    known = {"AMZ-TEST-1": Money(amount=50000, currency=Currency.INR)}
    outcome = validator.validate_offers(
        {"offers": [{
            "offer_id": "OFF-1", "product_id": "AMZ-NOPE",
            "kind": "upfront_discount",
            "value": {"amount": "1000", "currency": "INR"},
        }]},
        known_products=known,
    )
    assert outcome.accepted_count == 0
    assert outcome.quarantined[0][1] == "offer_for_unknown_product"


@pytest.mark.parametrize(
    ("offer", "expected"),
    [
        ({"kind": "unknown_kind", "value": {"amount": "10", "currency": "INR"}},
         "unknown_offer_kind"),
        ({"kind": "upfront_discount", "value": {"amount": "99999", "currency": "INR"}},
         "discount_exceeds_listed_price"),
        ({"kind": "upfront_discount", "value": {"amount": "48000", "currency": "INR"}},
         "implausible_discount_ratio"),
        ({"kind": "upfront_discount", "value": {"amount": "1000", "currency": "USD"}},
         "offer_currency_mismatch"),
        ({"kind": "upfront_discount", "percent": 150.0}, "invalid_offer_percent"),
        ({"kind": "upfront_discount"}, "offer_has_no_value"),
    ],
)
def test_quarantines_malformed_offers(validator, offer, expected):
    known = {"AMZ-TEST-1": Money(amount=50000, currency=Currency.INR)}
    payload = {"offer_id": "OFF-1", "product_id": "AMZ-TEST-1", **offer}
    outcome = validator.validate_offers({"offers": [payload]}, known_products=known)
    assert outcome.accepted_count == 0
    assert expected in outcome.quarantined[0][1]


def test_percentage_offer_is_capped_at_max_discount(validator):
    known = {"AMZ-TEST-1": Money(amount=50000, currency=Currency.INR)}
    outcome = validator.validate_offers(
        {"offers": [{
            "offer_id": "OFF-PCT", "product_id": "AMZ-TEST-1",
            "kind": "upfront_discount", "percent": 20.0,
            "max_discount": {"amount": "2000", "currency": "INR"},
        }]},
        known_products=known,
    )
    assert outcome.accepted_count == 1
    # 20% of 50000 is 10000, capped to the stated 2000 ceiling.
    assert outcome.accepted[0].value.amount == 2000
