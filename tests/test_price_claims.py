"""Screening model prose for invented monetary claims.

This module had **no test coverage**, which is why a live bug survived: the
percentage pattern ended in ``\\b`` after ``%``, and since ``%`` and the space
after it are both non-word characters, that boundary could never match. Every
``"Save 20% off"`` passed through the screen untouched, while the spelled-out
``"15 percent"`` was caught — so the module looked like it worked.
"""

from __future__ import annotations

import pytest

from laptop_agent.domain import Currency, Money
from laptop_agent.guardrails.price_claims import screen_price_claims


@pytest.fixture
def allowed() -> list[Money]:
    """What the deterministic pricing layer actually computed."""
    return [
        Money(amount="55000.00", currency=Currency.INR),  # effective
        Money(amount="60000.00", currency=Currency.INR),  # listed
        Money(amount="5000.00", currency=Currency.INR),   # savings
    ]


# --- percentages: never authorised in prose -------------------------------

@pytest.mark.parametrize(
    "prose",
    [
        "A 20% discount applies.",
        "Save 20% off the list price.",
        "That is 15 percent cheaper.",
        "Roughly 12.5% below the alternative.",
        "You get 30%off today.",
    ],
)
def test_discount_percentages_are_stripped(allowed, prose):
    report = screen_price_claims(prose, allowed)
    assert report.had_unauthorised_claims
    assert "%" not in report.text
    assert "percent" not in report.text.lower()


def test_stripped_percentage_leaves_readable_prose(allowed):
    """Substituting a phrase produced "A a discount discount applies."."""
    assert screen_price_claims("A 20% discount applies.", allowed).text == "A discount applies."
    assert screen_price_claims("That is 15 percent cheaper.", allowed).text == "That is cheaper."


# --- percentages that are specifications, not discounts ------------------

@pytest.mark.parametrize(
    "prose",
    [
        "It has 100% sRGB coverage.",
        "Covers 95% DCI-P3.",
        "Around 90% screen-to-body ratio.",
        "Reaches 72% NTSC colour gamut.",
        "Delivers 99% Adobe RGB.",
    ],
)
def test_display_specification_percentages_are_kept(allowed, prose):
    """A colour-gamut figure is a product fact. Removing it degrades a correct answer."""
    report = screen_price_claims(prose, allowed)
    assert not report.had_unauthorised_claims
    assert report.text == prose


# --- monetary amounts -----------------------------------------------------

@pytest.mark.parametrize(
    "prose",
    [
        "Saves you ₹12,000 versus the alternative.",
        "It costs Rs. 48,999 after offers.",
        "About 1.5 lakh all in.",
        "Only 80k for this configuration.",
        "Just $700 equivalent.",
    ],
)
def test_unauthorised_amounts_are_stripped(allowed, prose):
    report = screen_price_claims(prose, allowed)
    assert report.had_unauthorised_claims
    assert report.removed


@pytest.mark.parametrize(
    "prose",
    [
        "It costs ₹55,000.00 exactly.",
        "The list price is ₹60,000.00.",
    ],
)
def test_authorised_amounts_survive(allowed, prose):
    """A figure the pricing code computed is legitimate to restate."""
    report = screen_price_claims(prose, allowed)
    assert not report.had_unauthorised_claims
    assert report.text == prose


# --- specifications must never be mistaken for money ---------------------

@pytest.mark.parametrize(
    "prose",
    [
        "16GB RAM and a 512GB SSD.",
        "A 14 inch display at 165Hz.",
        "Weighs 1.4 kg with 9.5 hours of battery.",
        "1TB of storage and 8 cores.",
        "400 nits of brightness.",
        "It stays within your budget and is the cheaper option.",
    ],
)
def test_specifications_and_qualitative_language_are_untouched(allowed, prose):
    report = screen_price_claims(prose, allowed)
    assert not report.had_unauthorised_claims
    assert report.text == prose


# --- edge cases -----------------------------------------------------------

def test_empty_prose_is_safe(allowed):
    report = screen_price_claims("", allowed)
    assert report.text == ""
    assert not report.had_unauthorised_claims


def test_multiple_claims_are_all_reported(allowed):
    report = screen_price_claims(
        "Saves ₹12,000, a full 20% off, or about 1.2 lakh in total.", allowed
    )
    assert len(report.removed) >= 3


def test_no_allowed_amounts_strips_everything_monetary():
    """With nothing authorised, every figure is unauthorised."""
    report = screen_price_claims("It costs ₹55,000 after a 10% discount.", [])
    assert report.had_unauthorised_claims
    assert "55,000" not in report.text
    assert "%" not in report.text
