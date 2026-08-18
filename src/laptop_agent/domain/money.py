"""Money value object.

Prices are business-critical, so they are ``Decimal`` (never ``float``), always
carry a currency, and refuse cross-currency arithmetic. Every price in this
application originates from structured marketplace data and is represented by
this type — an LLM can never produce a ``Money``.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .enums import Currency

CENTS = Decimal("0.01")
# A laptop above this is almost certainly a malformed payload, not a product.
MAX_PLAUSIBLE_AMOUNT = Decimal("100000000")


class CurrencyMismatchError(ValueError):
    """Raised when arithmetic is attempted across two different currencies."""


class Money(BaseModel):
    """A non-negative monetary amount in a supported currency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    amount: Decimal
    currency: Currency

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, value: Any) -> Decimal:
        if isinstance(value, float):
            # Route floats through str so 0.1 does not become 0.1000000000000000055.
            value = str(value)
        if isinstance(value, (str, int)):
            try:
                value = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"not a valid monetary amount: {value!r}") from exc
        if not isinstance(value, Decimal):
            raise ValueError(f"not a valid monetary amount: {value!r}")
        return value

    @field_validator("amount")
    @classmethod
    def _validate_amount(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("monetary amount must be finite")
        if value < 0:
            raise ValueError("monetary amount cannot be negative")
        if value > MAX_PLAUSIBLE_AMOUNT:
            raise ValueError("monetary amount is implausibly large")
        return value.quantize(CENTS)

    # ----- constructors -----

    @classmethod
    def zero(cls, currency: Currency) -> Money:
        return cls(amount=Decimal("0"), currency=currency)

    # ----- arithmetic (currency-safe) -----

    def _assert_same_currency(self, other: Money) -> None:
        if self.currency is not other.currency:
            raise CurrencyMismatchError(
                f"cannot combine {self.currency} with {other.currency}"
            )

    def __add__(self, other: Money) -> Money:
        self._assert_same_currency(other)
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __sub__(self, other: Money) -> Money:
        """Subtract, clamping at zero.

        Clamping is deliberate: an effective price is never negative. Callers
        that need to detect over-subtraction compare magnitudes first — the
        :class:`~laptop_agent.guardrails.price_validator.PriceValidator` does
        exactly that before this is ever reached.
        """
        self._assert_same_currency(other)
        return Money(
            amount=max(Decimal("0"), self.amount - other.amount),
            currency=self.currency,
        )

    def __lt__(self, other: Money) -> bool:
        self._assert_same_currency(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        self._assert_same_currency(other)
        return self.amount <= other.amount

    def __gt__(self, other: Money) -> bool:
        self._assert_same_currency(other)
        return self.amount > other.amount

    def __ge__(self, other: Money) -> bool:
        self._assert_same_currency(other)
        return self.amount >= other.amount

    def percent_of(self, other: Money) -> Decimal:
        """This amount as a percentage of ``other`` (0 when ``other`` is zero)."""
        self._assert_same_currency(other)
        if other.amount == 0:
            return Decimal("0")
        return (self.amount / other.amount * Decimal("100")).quantize(CENTS)

    def __str__(self) -> str:
        symbol = "₹" if self.currency is Currency.INR else "$"
        return f"{symbol}{self.amount:,.2f}"


class MoneyInput(BaseModel):
    """Untrusted money shape as it arrives from a marketplace payload.

    Kept separate from :class:`Money` so a malformed external amount fails
    validation at the boundary instead of raising deep inside pricing code.
    """

    model_config = ConfigDict(extra="forbid")

    amount: Decimal | float | int | str
    currency: str

    @model_validator(mode="after")
    def _currency_supported(self) -> MoneyInput:
        if self.currency.upper() not in {c.value for c in Currency}:
            raise ValueError(f"unsupported currency: {self.currency!r}")
        return self

    def to_money(self) -> Money:
        return Money(amount=self.amount, currency=Currency(self.currency.upper()))
