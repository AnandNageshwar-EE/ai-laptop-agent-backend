"""The final gate. These tests attack the recommendation itself."""

from __future__ import annotations

from decimal import Decimal

import pytest

from laptop_agent.domain.enums import Bank, Currency, Marketplace, OfferKind, ProductCategory
from laptop_agent.domain.money import Money
from laptop_agent.domain.product import LaptopSpecs, Offer, Product, ProductCandidate
from laptop_agent.domain.recommendation import Recommendation
from laptop_agent.domain.requirements import LaptopRequirements, PurchaseProfile
from laptop_agent.domain.enums import UseCase
from laptop_agent.guardrails.price_validator import PriceValidator
from laptop_agent.guardrails.recommendation_validator import (
    ProviderRegistry,
    RecommendationValidator,
    ValidationFailure,
)
from laptop_agent.pricing.calculator import PriceCalculator
from laptop_agent.ranking.scorer import rank_candidates, score_candidate

INR = Currency.INR


def money(amount, currency: Currency = INR) -> Money:
    return Money(amount=amount, currency=currency)


def product(
    product_id: str = "AMZ-GOOD-1",
    price: str = "60000",
    url: str = "https://www.amazon.in/dp/GOOD1",
    ram: int = 16,
    in_stock: bool = True,
) -> Product:
    return Product(
        product_id=product_id,
        marketplace=Marketplace.AMAZON,
        category=ProductCategory.LAPTOP,
        title=f"Laptop {product_id}",
        brand="test",
        url=url,
        listed_price=money(price),
        mrp=money("75000"),
        rating=4.3,
        rating_count=800,
        in_stock=in_stock,
        specs=LaptopSpecs(
            ram_gb=ram, storage_gb=512, storage_type="ssd", cpu="Test CPU",
            screen_inches=14.0, weight_kg=1.4, battery_hours=9.0, os="windows",
        ),
    )


@pytest.fixture
def requirements() -> LaptopRequirements:
    return LaptopRequirements(
        use_case=UseCase.SOFTWARE_DEVELOPMENT,
        budget_max=money("80000"),
        min_ram_gb=16,
        mandatory_fields=["budget_max", "min_ram_gb"],
    )


@pytest.fixture
def setup(requirements):
    """A clean, valid pipeline result the tests then tamper with."""
    products = [product(), product("AMZ-GOOD-2", price="70000", url="https://www.amazon.in/dp/GOOD2")]
    offers: list[Offer] = []
    profile = PurchaseProfile()
    candidates, _ = PriceCalculator().build_candidates(products, offers, requirements, profile)
    ranked = rank_candidates(candidates, requirements)
    registry = ProviderRegistry()
    registry.register_all(products)

    winner = next(c for c in candidates if c.key == ranked[0].key)
    recommendation = Recommendation(
        product_id=winner.product.product_id,
        marketplace=winner.product.marketplace,
        title=winner.product.title,
        url=winner.product.url,
        listed_price=winner.price.listed_price,
        effective_price=winner.price.effective_price,
        upfront_savings=winner.price.total_upfront_discount,
        cashback_value=winner.price.cashback_value,
        score=ranked[0].total,
        scoring_version=ranked[0].scoring_version,
        rationale="This laptop meets the stated requirements for development work.",
    )
    return {
        "candidates": candidates,
        "ranked": ranked,
        "registry": registry,
        "requirements": requirements,
        "profile": profile,
        "recommendation": recommendation,
        "products": products,
    }


@pytest.fixture
def validator() -> RecommendationValidator:
    return RecommendationValidator(PriceValidator())


def validate(validator, setup, **overrides):
    kwargs = {
        "recommendation": setup["recommendation"],
        "candidates": setup["candidates"],
        "ranked": setup["ranked"],
        "requirements": setup["requirements"],
        "profile": setup["profile"],
        "registry": setup["registry"],
        "rescore": score_candidate,
    }
    kwargs.update(overrides)
    return validator.validate(**kwargs)


def test_valid_recommendation_passes(validator, setup):
    result = validate(validator, setup)
    assert result.is_valid, result.failures


def test_missing_recommendation_fails(validator, setup):
    result = validate(validator, setup, recommendation=None)
    assert ValidationFailure.NO_RECOMMENDATION in result.failures


def test_candidate_not_in_candidate_set_fails(validator, setup):
    fabricated = setup["recommendation"].model_copy(update={"product_id": "AMZ-FAKE-1"})
    result = validate(validator, setup, recommendation=fabricated)
    assert ValidationFailure.CANDIDATE_NOT_FOUND in result.failures


def test_candidate_not_from_provider_fails(validator, setup):
    """A candidate present in the candidate set but absent from provider data."""
    ghost_product = product("AMZ-GHOST-1", url="https://www.amazon.in/dp/GHOST")
    ghost_price, _ = PriceValidator().compute(ghost_product, [], setup["profile"])
    ghost = ProductCandidate(
        product=ghost_product,
        offers=[],
        price=ghost_price,
        hard_requirements_passed=True,
    )
    ghost_score = score_candidate(ghost, setup["requirements"])
    fabricated = setup["recommendation"].model_copy(
        update={
            "product_id": "AMZ-GHOST-1",
            "title": ghost_product.title,
            "url": ghost_product.url,
            "listed_price": ghost_price.listed_price,
            "effective_price": ghost_price.effective_price,
            "upfront_savings": ghost_price.total_upfront_discount,
            "score": ghost_score.total,
        }
    )
    result = validate(
        validator,
        setup,
        recommendation=fabricated,
        candidates=[*setup["candidates"], ghost],
        ranked=[*setup["ranked"], ghost_score],
    )
    assert ValidationFailure.NOT_FROM_PROVIDER in result.failures


def test_swapped_url_fails(validator, setup):
    tampered = setup["recommendation"].model_copy(
        update={"url": "https://www.amazon.in/dp/SOMETHING-ELSE"}
    )
    result = validate(validator, setup, recommendation=tampered)
    assert ValidationFailure.URL_NOT_FROM_PROVIDER in result.failures


def test_offsite_url_fails_on_host_check(validator, setup):
    tampered = setup["recommendation"].model_copy(
        update={"url": "https://phishing.example.com/dp/GOOD1"}
    )
    result = validate(validator, setup, recommendation=tampered)
    assert ValidationFailure.URL_UNTRUSTED_HOST in result.failures
    assert ValidationFailure.URL_NOT_FROM_PROVIDER in result.failures


def test_altered_title_fails(validator, setup):
    tampered = setup["recommendation"].model_copy(
        update={"title": "Laptop AMZ-GOOD-1 (BEST DEAL EVER)"}
    )
    result = validate(validator, setup, recommendation=tampered)
    assert ValidationFailure.FIELD_MISMATCH in result.failures


def test_understated_price_fails_reproducibility(validator, setup):
    """The classic attack: report a lower price than the provider quoted."""
    original = setup["recommendation"]
    tampered = original.model_copy(
        update={
            "effective_price": money("1000"),
            "upfront_savings": original.listed_price - money("1000"),
        }
    )
    result = validate(validator, setup, recommendation=tampered)
    assert ValidationFailure.PRICE_NOT_REPRODUCIBLE in result.failures


def test_inflated_savings_fails(validator, setup):
    original = setup["recommendation"]
    tampered = original.model_copy(
        update={
            "upfront_savings": money("20000"),
            "effective_price": original.listed_price - money("20000"),
        }
    )
    result = validate(validator, setup, recommendation=tampered)
    assert ValidationFailure.PRICE_NOT_REPRODUCIBLE in result.failures


def test_cashback_presented_as_discount_fails(validator, setup):
    original = setup["recommendation"]
    tampered = original.model_copy(update={"cashback_value": money("5000")})
    result = validate(validator, setup, recommendation=tampered)
    assert ValidationFailure.PRICE_NOT_REPRODUCIBLE in result.failures


def test_tampered_score_fails(validator, setup):
    tampered = setup["recommendation"].model_copy(update={"score": Decimal("0.999999")})
    result = validate(validator, setup, recommendation=tampered)
    assert ValidationFailure.SCORE_NOT_REPRODUCIBLE in result.failures


def test_candidate_violating_a_mandatory_constraint_fails(validator, requirements):
    """8GB RAM against a mandatory 16GB minimum must never be recommended."""
    weak = product("AMZ-WEAK-1", ram=8, url="https://www.amazon.in/dp/WEAK1")
    profile = PurchaseProfile()
    price, _ = PriceValidator().compute(weak, [], profile)
    candidate = ProductCandidate(
        product=weak,
        offers=[],
        price=price,
        hard_requirements_passed=False,
        failed_constraints=["min_ram_gb"],
    )
    score = score_candidate(candidate, requirements)
    registry = ProviderRegistry()
    registry.register_all([weak])
    recommendation = Recommendation(
        product_id=weak.product_id,
        marketplace=weak.marketplace,
        title=weak.title,
        url=weak.url,
        listed_price=price.listed_price,
        effective_price=price.effective_price,
        upfront_savings=price.total_upfront_discount,
        cashback_value=price.cashback_value,
        score=score.total,
        scoring_version=score.scoring_version,
        rationale="Attempting to recommend an under-specified laptop.",
    )
    result = RecommendationValidator(PriceValidator()).validate(
        recommendation=recommendation,
        candidates=[candidate],
        ranked=[score],
        requirements=requirements,
        profile=profile,
        registry=registry,
        rescore=score_candidate,
    )
    assert not result.is_valid
    assert ValidationFailure.HARD_REQUIREMENT_FAILED in result.failures
    assert ValidationFailure.MANDATORY_CONSTRAINT_VIOLATED in result.failures


def test_out_of_stock_candidate_fails(validator, setup, requirements):
    oos = product("AMZ-OOS-1", in_stock=False, url="https://www.amazon.in/dp/OOS1")
    profile = setup["profile"]
    price, _ = PriceValidator().compute(oos, [], profile)
    candidate = ProductCandidate(
        product=oos, offers=[], price=price, hard_requirements_passed=True
    )
    score = score_candidate(candidate, requirements)
    registry = ProviderRegistry()
    registry.register_all([oos])
    recommendation = setup["recommendation"].model_copy(
        update={
            "product_id": oos.product_id,
            "title": oos.title,
            "url": oos.url,
            "listed_price": price.listed_price,
            "effective_price": price.effective_price,
            "upfront_savings": price.total_upfront_discount,
            "score": score.total,
        }
    )
    result = validate(
        validator,
        setup,
        recommendation=recommendation,
        candidates=[candidate],
        ranked=[score],
        registry=registry,
    )
    assert ValidationFailure.OUT_OF_STOCK in result.failures


def test_not_top_ranked_candidate_fails(validator, setup):
    """Recommending a lower-ranked candidate is rejected."""
    ranked = setup["ranked"]
    if len(ranked) < 2:
        pytest.skip("needs at least two candidates")
    second = next(c for c in setup["candidates"] if c.key == ranked[1].key)
    recommendation = setup["recommendation"].model_copy(
        update={
            "product_id": second.product.product_id,
            "title": second.product.title,
            "url": second.product.url,
            "listed_price": second.price.listed_price,
            "effective_price": second.price.effective_price,
            "upfront_savings": second.price.total_upfront_discount,
            "score": ranked[1].total,
        }
    )
    result = validate(validator, setup, recommendation=recommendation)
    assert ValidationFailure.NOT_IN_RANKED_SET in result.failures


def test_recommendation_model_rejects_incoherent_prices():
    """Arithmetic is enforced by the model, so a bad triple cannot exist."""
    with pytest.raises(Exception):
        Recommendation(
            product_id="AMZ-X1",
            marketplace=Marketplace.AMAZON,
            title="X",
            url="https://www.amazon.in/dp/X1",
            listed_price=money("50000"),
            effective_price=money("40000"),
            upfront_savings=money("5000"),  # does not close: 50000-5000 != 40000
            cashback_value=money("0"),
            score=Decimal("0.5"),
            scoring_version="v1",
            rationale="Incoherent pricing should be impossible to construct.",
        )
