"""Recommendation model.

Every price-bearing field here is copied from a validated ``PriceBreakdown``.
The LLM contributes ``rationale`` and ``trade_offs`` prose only, and even that
is screened for invented monetary figures before it reaches this model
(see :mod:`laptop_agent.guardrails.price_claims`).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .enums import Marketplace
from .money import Money
from .product import AppliedOffer, LaptopSpecs


class TradeOff(BaseModel):
    """A concession the user makes by choosing this product."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: Annotated[str, Field(min_length=2, max_length=48)]
    detail: Annotated[str, Field(min_length=2, max_length=280)]


class RunnerUp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: str
    marketplace: Marketplace
    title: Annotated[str, Field(max_length=300)]
    brand: Annotated[str, Field(max_length=32)] = ""
    url: HttpUrl
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    rating_count: Annotated[int, Field(ge=0)] = 0
    listed_price: Money
    effective_price: Money
    upfront_savings: Money
    cashback_value: Money
    unmet_conditional_offers: list[str] = Field(default_factory=list)
    #: Copied from the validated provider product, so each result can be rendered
    #: as its own card without the UI re-deriving anything.
    specs: LaptopSpecs
    score: Annotated[Decimal, Field(ge=0, le=1)]
    why_not: Annotated[str, Field(max_length=280)] = ""

    @model_validator(mode="after")
    def _price_coherence(self) -> RunnerUp:
        if (
            self.listed_price.amount - self.upfront_savings.amount
            != self.effective_price.amount
        ):
            raise ValueError("effective_price must equal listed_price - upfront_savings")
        return self


class Recommendation(BaseModel):
    """The final, validator-approved answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: str
    marketplace: Marketplace
    title: Annotated[str, Field(max_length=300)]
    brand: Annotated[str, Field(max_length=32)] = ""
    #: Copied verbatim from the provider-returned product. Never LLM-authored.
    url: HttpUrl
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    rating_count: Annotated[int, Field(ge=0)] = 0
    #: Copied from the validated provider product. The validator re-checks these
    #: against provider data, so the UI can display them as authoritative.
    specs: LaptopSpecs

    listed_price: Money
    effective_price: Money
    upfront_savings: Money
    cashback_value: Money
    #: Offers the user would additionally get if they qualify. Never assumed.
    unmet_conditional_offers: list[str] = Field(default_factory=list)
    #: The discounts actually subtracted, so the price can be shown as a
    #: breakdown rather than a single unexplained number.
    applied_offers: list[AppliedOffer] = Field(default_factory=list)

    score: Annotated[Decimal, Field(ge=0, le=1)]
    scoring_version: str

    rationale: Annotated[str, Field(min_length=10, max_length=1200)]
    trade_offs: Annotated[list[TradeOff], Field(max_length=6)] = Field(default_factory=list)
    runner_ups: Annotated[list[RunnerUp], Field(max_length=5)] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _price_coherence(self) -> Recommendation:
        currencies = {
            self.listed_price.currency,
            self.effective_price.currency,
            self.upfront_savings.currency,
            self.cashback_value.currency,
        }
        if len(currencies) != 1:
            raise ValueError("recommendation amounts must share one currency")
        if self.effective_price > self.listed_price:
            raise ValueError("effective_price cannot exceed listed_price")
        if (
            self.listed_price.amount - self.upfront_savings.amount
            != self.effective_price.amount
        ):
            raise ValueError("effective_price must equal listed_price - upfront_savings")
        return self

    @property
    def key(self) -> tuple[Marketplace, str]:
        return (self.marketplace, self.product_id)
