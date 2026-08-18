"""Products, offers and priced candidates.

These are the *validated* shapes. Raw marketplace JSON never becomes state —
it is parsed into ``ProductPayload``/``OfferPayload`` at the boundary
(see :mod:`laptop_agent.guardrails.tool_output`), and only the resulting
``Product``/``Offer`` instances enter the graph.

``Product.description`` holds untrusted marketplace prose. It is carried as
*data*: it is never concatenated into a system prompt, and when shown to the
model it is wrapped by :mod:`laptop_agent.guardrails.untrusted`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from .enums import Bank, Currency, Marketplace, OfferKind, ProductCategory
from .money import Money
from .requirements import LaptopRequirements, PurchaseProfile

PRODUCT_ID_PATTERN = r"^[A-Za-z0-9._-]{4,64}$"


class LaptopSpecs(BaseModel):
    """Structured specification. Only these fields are used for hard-constraint
    checks — free text is never parsed to decide whether a requirement passes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ram_gb: Annotated[int, Field(ge=2, le=256)]
    storage_gb: Annotated[int, Field(ge=32, le=8192)]
    storage_type: Literal["ssd", "hdd"]
    cpu: Annotated[str, Field(min_length=2, max_length=64)]
    gpu: Annotated[str, Field(max_length=64)] = ""
    dedicated_gpu: bool = False
    screen_inches: Annotated[float, Field(ge=8.0, le=20.0)]
    weight_kg: Annotated[float, Field(gt=0.2, le=6.0)]
    battery_hours: Annotated[float, Field(ge=1.0, le=30.0)]
    os: Literal["windows", "macos", "linux", "chromeos"]
    touchscreen: bool = False
    refresh_rate_hz: Annotated[int, Field(ge=30, le=480)] = 60


class Product(BaseModel):
    """A product as returned by a marketplace provider and validated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: Annotated[str, Field(pattern=PRODUCT_ID_PATTERN)]
    marketplace: Marketplace
    category: ProductCategory
    title: Annotated[str, Field(min_length=3, max_length=300)]
    brand: Annotated[str, Field(min_length=1, max_length=32)]
    url: HttpUrl
    listed_price: Money
    mrp: Money | None = None
    rating: Annotated[float, Field(ge=0.0, le=5.0)] | None = None
    rating_count: Annotated[int, Field(ge=0)] = 0
    in_stock: bool = True
    specs: LaptopSpecs
    #: Untrusted marketplace prose. Data, never instruction.
    description: Annotated[str, Field(max_length=2000)] = ""

    @field_validator("brand")
    @classmethod
    def _normalise_brand(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def _price_coherence(self) -> Product:
        if self.mrp is not None:
            if self.mrp.currency is not self.listed_price.currency:
                raise ValueError("mrp and listed_price must share a currency")
            if self.listed_price > self.mrp:
                raise ValueError("listed_price cannot exceed mrp")
        return self

    @property
    def key(self) -> tuple[Marketplace, str]:
        """Provenance key used by the recommendation validator."""
        return (self.marketplace, self.product_id)

    @property
    def currency(self) -> Currency:
        return self.listed_price.currency


class Offer(BaseModel):
    """A validated offer attached to a specific product on a specific marketplace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    offer_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9._-]{4,64}$")]
    marketplace: Marketplace
    product_id: Annotated[str, Field(pattern=PRODUCT_ID_PATTERN)]
    kind: OfferKind
    #: Absolute value. Percentage offers are resolved to an absolute amount by
    #: the marketplace adapter, capped by ``max_discount``, before validation.
    value: Money
    description: Annotated[str, Field(max_length=500)] = ""
    requires_bank: Bank | None = None
    requires_exchange: bool = False
    min_transaction: Money | None = None
    stackable: bool = True

    @model_validator(mode="after")
    def _coherence(self) -> Offer:
        if self.kind is OfferKind.NO_COST_EMI and self.value.amount != 0:
            raise ValueError("no_cost_emi changes financing terms, not price; value must be 0")
        if self.kind is OfferKind.BANK_DISCOUNT and self.requires_bank is None:
            raise ValueError("bank_discount must declare requires_bank")
        if self.kind is OfferKind.EXCHANGE_BONUS and not self.requires_exchange:
            raise ValueError("exchange_bonus must declare requires_exchange")
        if self.min_transaction and self.min_transaction.currency is not self.value.currency:
            raise ValueError("min_transaction and value must share a currency")
        return self

    @property
    def key(self) -> tuple[Marketplace, str]:
        return (self.marketplace, self.product_id)

    def applies_to(self, profile: PurchaseProfile, listed_price: Money) -> bool:
        """Whether this offer's conditions are actually met.

        A conditional offer that is not met is *never* silently treated as
        guaranteed — it is reported separately as "available if you qualify".
        """
        if not self.kind.reduces_upfront_price:
            return False
        if self.requires_bank is not None and not profile.is_eligible_for(self.requires_bank):
            return False
        if self.requires_exchange and not profile.has_exchange_device:
            return False
        if self.min_transaction is not None and listed_price < self.min_transaction:
            return False
        return True


class AppliedOffer(BaseModel):
    """An offer that was actually subtracted, with the amount subtracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    offer_id: str
    kind: OfferKind
    amount: Money
    reason: Annotated[str, Field(max_length=200)] = ""


class PriceBreakdown(BaseModel):
    """Deterministic, auditable price computation.

    ``effective_price`` is what the user pays at checkout. Cashback is tracked
    separately in ``cashback_value`` and is **never** subtracted from it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    listed_price: Money
    applied_offers: list[AppliedOffer] = Field(default_factory=list)
    total_upfront_discount: Money
    effective_price: Money
    #: Returned after purchase, not a checkout reduction.
    cashback_value: Money
    #: Offer ids that exist but whose conditions are not met by this user.
    unmet_conditional_offers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_valid: bool = True

    @model_validator(mode="after")
    def _arithmetic_holds(self) -> PriceBreakdown:
        currencies = {
            self.listed_price.currency,
            self.total_upfront_discount.currency,
            self.effective_price.currency,
            self.cashback_value.currency,
        }
        if len(currencies) != 1:
            raise ValueError("all amounts in a breakdown must share one currency")
        expected = self.listed_price.amount - self.total_upfront_discount.amount
        if self.effective_price.amount != expected:
            raise ValueError(
                "effective_price must equal listed_price - total_upfront_discount"
            )
        summed = sum((o.amount.amount for o in self.applied_offers), start=Decimal("0"))
        if summed != self.total_upfront_discount.amount:
            raise ValueError("total_upfront_discount must equal the sum of applied offers")
        return self

    @property
    def savings(self) -> Money:
        return self.total_upfront_discount


class ProductCandidate(BaseModel):
    """A product plus its offers, its computed price and its constraint verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product: Product
    offers: list[Offer] = Field(default_factory=list)
    price: PriceBreakdown
    hard_requirements_passed: bool
    failed_constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verdict_matches_reasons(self) -> ProductCandidate:
        if self.hard_requirements_passed and self.failed_constraints:
            raise ValueError("candidate cannot both pass and list failed constraints")
        if not self.hard_requirements_passed and not self.failed_constraints:
            raise ValueError("a failing candidate must name the failed constraints")
        return self

    @property
    def key(self) -> tuple[Marketplace, str]:
        return self.product.key

    @property
    def is_recommendable(self) -> bool:
        return (
            self.hard_requirements_passed
            and self.price.is_valid
            and self.product.in_stock
        )


def evaluate_hard_constraints(
    product: Product,
    requirements: LaptopRequirements,
    price: PriceBreakdown,
) -> list[str]:
    """Return the names of mandatory constraints this product violates.

    Deterministic and side-effect free, so the recommendation validator can
    re-run it independently of the node that produced the candidate.
    """
    specs = product.specs
    failures: list[str] = []

    def fails(field: str) -> None:
        if field in requirements.mandatory_fields:
            failures.append(field)

    if requirements.budget_max is not None:
        if price.effective_price.currency is not requirements.budget_max.currency:
            failures.append("budget_max:currency_mismatch")
        elif price.effective_price > requirements.budget_max:
            fails("budget_max")
    if requirements.min_ram_gb is not None and specs.ram_gb < requirements.min_ram_gb:
        fails("min_ram_gb")
    if requirements.min_storage_gb is not None and specs.storage_gb < requirements.min_storage_gb:
        fails("min_storage_gb")
    if requirements.storage_type != "any" and specs.storage_type != requirements.storage_type:
        fails("storage_type")
    if requirements.min_screen_inches is not None and specs.screen_inches < requirements.min_screen_inches:
        fails("min_screen_inches")
    if requirements.max_screen_inches is not None and specs.screen_inches > requirements.max_screen_inches:
        fails("max_screen_inches")
    if requirements.max_weight_kg is not None and specs.weight_kg > requirements.max_weight_kg:
        fails("max_weight_kg")
    if requirements.min_battery_hours is not None and specs.battery_hours < requirements.min_battery_hours:
        fails("min_battery_hours")
    if requirements.required_os != "any" and specs.os != requirements.required_os:
        fails("required_os")
    if requirements.dedicated_gpu_required and not specs.dedicated_gpu:
        fails("dedicated_gpu_required")
    if requirements.touchscreen_required and not specs.touchscreen:
        fails("touchscreen_required")
    if requirements.preferred_brands and product.brand not in requirements.preferred_brands:
        fails("preferred_brands")
    if product.brand in requirements.excluded_brands:
        fails("excluded_brands")

    return sorted(set(failures))
