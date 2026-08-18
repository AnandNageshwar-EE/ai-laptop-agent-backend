"""Closed vocabularies.

Anything that crosses a trust boundary (LLM output, marketplace payload, tool
argument) is constrained to these enums, so an unexpected string is a validation
error rather than data that silently flows into the graph state.
"""

from __future__ import annotations

from enum import StrEnum


class Marketplace(StrEnum):
    AMAZON = "amazon"
    FLIPKART = "flipkart"


class Currency(StrEnum):
    INR = "INR"
    USD = "USD"


class ProductCategory(StrEnum):
    LAPTOP = "laptop"
    LAPTOP_ACCESSORY = "laptop_accessory"


class UseCase(StrEnum):
    GENERAL = "general"
    STUDENT = "student"
    OFFICE_PRODUCTIVITY = "office_productivity"
    SOFTWARE_DEVELOPMENT = "software_development"
    DATA_SCIENCE = "data_science"
    GAMING = "gaming"
    CONTENT_CREATION = "content_creation"


class OfferKind(StrEnum):
    """How an offer affects the price.

    The distinction is business-critical:

    * ``UPFRONT_DISCOUNT`` reduces the amount paid at checkout, unconditionally.
    * ``BANK_DISCOUNT`` / ``EXCHANGE_BONUS`` / ``COUPON`` reduce the amount paid
      only when an eligibility condition holds. They are never assumed.
    * ``CASHBACK`` does *not* reduce the amount paid at checkout. It is returned
      later and must never be subtracted from the effective price.
    * ``NO_COST_EMI`` changes financing terms, not the price. Value is always 0.
    """

    UPFRONT_DISCOUNT = "upfront_discount"
    BANK_DISCOUNT = "bank_discount"
    EXCHANGE_BONUS = "exchange_bonus"
    COUPON = "coupon"
    CASHBACK = "cashback"
    NO_COST_EMI = "no_cost_emi"

    @property
    def is_conditional(self) -> bool:
        return self in {
            OfferKind.BANK_DISCOUNT,
            OfferKind.EXCHANGE_BONUS,
            OfferKind.COUPON,
        }

    @property
    def reduces_upfront_price(self) -> bool:
        """True only for offers that lower the checkout amount."""
        return self in {
            OfferKind.UPFRONT_DISCOUNT,
            OfferKind.BANK_DISCOUNT,
            OfferKind.EXCHANGE_BONUS,
            OfferKind.COUPON,
        }


class Bank(StrEnum):
    HDFC = "hdfc"
    ICICI = "icici"
    SBI = "sbi"
    AXIS = "axis"
    KOTAK = "kotak"


class RejectionReason(StrEnum):
    EMPTY_INPUT = "empty_input"
    INPUT_TOO_LONG = "input_too_long"
    MALFORMED_INPUT = "malformed_input"
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_MANIPULATION = "system_manipulation"
    SECRET_EXFILTRATION = "secret_exfiltration"
    DISALLOWED_TOPIC = "disallowed_topic"
    OUT_OF_SCOPE = "out_of_scope"
