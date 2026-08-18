"""Price guardrails. Prices are business-critical, so these are the strictest tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from laptop_agent.domain.enums import Bank, Currency, Marketplace, OfferKind, ProductCategory
from laptop_agent.domain.money import CurrencyMismatchError, Money
from laptop_agent.domain.product import LaptopSpecs, Offer, Product
from laptop_agent.domain.requirements import PurchaseProfile
from laptop_agent.guardrails.price_validator import PriceValidator

INR = Currency.INR


def money(amount: str | int, currency: Currency = INR) -> Money:
    return Money(amount=amount, currency=currency)


def make_product(price: str = "50000", mrp: str | None = "60000") -> Product:
    return Product(
        product_id="AMZ-P1",
        marketplace=Marketplace.AMAZON,
        category=ProductCategory.LAPTOP,
        title="Test Laptop",
        brand="test",
        url="https://www.amazon.in/dp/P1",
        listed_price=money(price),
        mrp=money(mrp) if mrp else None,
        specs=LaptopSpecs(
            ram_gb=16, storage_gb=512, storage_type="ssd", cpu="Test",
            screen_inches=14.0, weight_kg=1.4, battery_hours=8.0, os="windows",
        ),
    )


def make_offer(
    offer_id: str,
    kind: OfferKind,
    value: str,
    *,
    bank: Bank | None = None,
    exchange: bool = False,
    min_transaction: str | None = None,
    stackable: bool = True,
) -> Offer:
    return Offer(
        offer_id=offer_id,
        marketplace=Marketplace.AMAZON,
        product_id="AMZ-P1",
        kind=kind,
        value=money(value),
        requires_bank=bank,
        requires_exchange=exchange,
        min_transaction=money(min_transaction) if min_transaction else None,
        stackable=stackable,
    )


@pytest.fixture
def validator() -> PriceValidator:
    return PriceValidator()


# --- money type ------------------------------------------------------------

def test_money_rejects_negative_and_nonfinite():
    for bad in ("-1", "-0.01", "NaN", "Infinity"):
        with pytest.raises(Exception):
            money(bad)


def test_money_refuses_cross_currency_arithmetic():
    with pytest.raises(CurrencyMismatchError):
        money(100, Currency.INR) + money(100, Currency.USD)


def test_money_uses_decimal_not_float():
    # 0.1 + 0.2 must be exactly 0.30, which float arithmetic cannot promise.
    assert (money("0.1") + money("0.2")).amount == Decimal("0.30")


# --- product-level checks --------------------------------------------------

def test_rejects_listed_price_above_mrp(validator):
    with pytest.raises(Exception):
        make_product(price="70000", mrp="60000")


def test_rejects_implausible_mrp_discount(validator):
    product = make_product(price="29990", mrp="400000")  # 92.5% gap
    report = validator.validate_product_price(product)
    assert not report.is_valid
    assert "implausible_mrp_discount" in report.errors


# --- the eight required checks --------------------------------------------

def test_upfront_discount_reduces_price(validator, empty_profile):
    breakdown, report = validator.compute(
        make_product(), [make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "5000")], empty_profile
    )
    assert report.is_valid
    assert breakdown.effective_price == money("45000")
    assert breakdown.total_upfront_discount == money("5000")


def test_cashback_is_never_subtracted_from_checkout_price(validator, empty_profile):
    breakdown, report = validator.compute(
        make_product(), [make_offer("OFF-01", OfferKind.CASHBACK, "3000")], empty_profile
    )
    assert report.is_valid
    # The whole point: the user still pays the full listed price today.
    assert breakdown.effective_price == money("50000")
    assert breakdown.total_upfront_discount == money("0")
    assert breakdown.cashback_value == money("3000")
    assert "cashback_excluded_from_effective_price" in breakdown.warnings


def test_conditional_bank_offer_not_applied_without_eligibility(validator, empty_profile):
    offer = make_offer("OFF-01", OfferKind.BANK_DISCOUNT, "4000", bank=Bank.HDFC)
    breakdown, report = validator.compute(make_product(), [offer], empty_profile)
    assert report.is_valid
    assert breakdown.effective_price == money("50000")
    assert breakdown.unmet_conditional_offers == ["OFF-01"]


def test_conditional_bank_offer_applied_when_eligible(validator, hdfc_profile):
    offer = make_offer("OFF-01", OfferKind.BANK_DISCOUNT, "4000", bank=Bank.HDFC)
    breakdown, report = validator.compute(make_product(), [offer], hdfc_profile)
    assert report.is_valid
    assert breakdown.effective_price == money("46000")
    assert breakdown.unmet_conditional_offers == []


def test_bank_offer_respects_minimum_transaction(validator, hdfc_profile):
    offer = make_offer(
        "OFF-01", OfferKind.BANK_DISCOUNT, "4000", bank=Bank.HDFC, min_transaction="60000"
    )
    breakdown, _ = validator.compute(make_product(price="50000"), [offer], hdfc_profile)
    assert breakdown.effective_price == money("50000")
    assert breakdown.unmet_conditional_offers == ["OFF-01"]


def test_exchange_bonus_requires_a_device(validator, empty_profile):
    offer = make_offer("OFF-01", OfferKind.EXCHANGE_BONUS, "4000", exchange=True)
    breakdown, _ = validator.compute(make_product(), [offer], empty_profile)
    assert breakdown.effective_price == money("50000")

    with_device = PurchaseProfile(has_exchange_device=True)
    breakdown, _ = validator.compute(make_product(), [offer], with_device)
    assert breakdown.effective_price == money("46000")


def test_duplicate_offer_id_is_not_applied_twice(validator, empty_profile):
    offer = make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "5000")
    breakdown, report = validator.compute(make_product(), [offer, offer], empty_profile)
    assert report.is_valid
    assert breakdown.effective_price == money("45000")
    assert len(breakdown.applied_offers) == 1


def test_same_discount_under_two_ids_is_not_applied_twice(validator, empty_profile):
    """Two records for one promotion is how a double discount happens in practice."""
    breakdown, report = validator.compute(
        make_product(),
        [
            make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "5000"),
            make_offer("OFF-02", OfferKind.UPFRONT_DISCOUNT, "5000"),
        ],
        empty_profile,
    )
    assert report.is_valid
    assert breakdown.effective_price == money("45000")
    assert any("duplicate_discount_ignored" in w for w in breakdown.warnings)


def test_no_cost_emi_does_not_change_price(validator, empty_profile):
    breakdown, report = validator.compute(
        make_product(), [make_offer("OFF-01", OfferKind.NO_COST_EMI, "0")], empty_profile
    )
    assert report.is_valid
    assert breakdown.effective_price == money("50000")


def test_effective_price_never_negative(validator, empty_profile):
    breakdown, report = validator.compute(
        make_product(price="10000"),
        [make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "9999999")],
        empty_profile,
    )
    assert not report.is_valid
    assert not breakdown.is_valid
    assert breakdown.effective_price.amount >= 0
    # An unusable price is never turned into an attractive one.
    assert breakdown.effective_price == money("10000")


def test_stacked_discounts_cannot_exceed_listed_price(validator, hdfc_profile):
    breakdown, report = validator.compute(
        make_product(price="50000"),
        [
            make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "30000"),
            make_offer("OFF-02", OfferKind.BANK_DISCOUNT, "25000", bank=Bank.HDFC),
        ],
        hdfc_profile,
    )
    assert not report.is_valid
    assert not breakdown.is_valid


def test_unsupported_currency_is_rejected(validator, empty_profile):
    product = make_product().model_copy(
        update={"listed_price": money("1000", Currency.USD), "mrp": None}
    )
    offer = make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "100")  # INR
    _, report = validator.compute(product, [offer], empty_profile)
    assert not report.is_valid
    assert any("currency_mismatch" in e for e in report.errors)


def test_offer_for_a_different_product_is_rejected(validator, empty_profile):
    foreign = make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "1000").model_copy(
        update={"product_id": "AMZ-OTHER"}
    )
    _, report = validator.compute(make_product(), [foreign], empty_profile)
    assert not report.is_valid
    assert any("offer_product_mismatch" in e for e in report.errors)


def test_breakdown_arithmetic_is_self_checking():
    """The model refuses to hold a breakdown whose arithmetic does not close."""
    from laptop_agent.domain.product import AppliedOffer, PriceBreakdown

    with pytest.raises(Exception):
        PriceBreakdown(
            listed_price=money("50000"),
            applied_offers=[
                AppliedOffer(offer_id="OFF-01", kind=OfferKind.UPFRONT_DISCOUNT, amount=money("5000"))
            ],
            total_upfront_discount=money("5000"),
            effective_price=money("40000"),  # should be 45000
            cashback_value=money("0"),
        )


def test_computation_is_order_independent(validator, hdfc_profile):
    offers = [
        make_offer("OFF-01", OfferKind.UPFRONT_DISCOUNT, "3000"),
        make_offer("OFF-02", OfferKind.BANK_DISCOUNT, "2000", bank=Bank.HDFC),
        make_offer("OFF-03", OfferKind.CASHBACK, "1000"),
    ]
    first, _ = validator.compute(make_product(), offers, hdfc_profile)
    second, _ = validator.compute(make_product(), list(reversed(offers)), hdfc_profile)
    # Reproducibility is what lets the recommendation validator recompute and compare.
    assert first.effective_price == second.effective_price
    assert first.cashback_value == second.cashback_value
    assert [o.offer_id for o in first.applied_offers] == [
        o.offer_id for o in second.applied_offers
    ]
