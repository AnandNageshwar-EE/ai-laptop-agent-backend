"""Price validation and the deterministic effective-price computation.

No LLM participates in any part of this file. Prices enter only from validated
marketplace data and leave only as a :class:`PriceBreakdown` whose arithmetic is
self-checking.

The checks, in the order the spec lists them:

* ``price > 0``
* ``discount >= 0``
* ``discount <= listed price``
* ``effective price >= 0``
* currency is supported and consistent throughout
* no duplicate discounts — the same offer, or the same offer *kind* from a
  duplicate record, cannot be subtracted twice
* conditional discounts are not treated as guaranteed
* cashback is not treated as an upfront discount

A product whose price data fails any check is marked invalid and is never
recommended.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import Currency, OfferKind
from ..domain.money import Money
from ..domain.product import AppliedOffer, Offer, PriceBreakdown, Product
from ..domain.requirements import PurchaseProfile

#: Currencies this application is prepared to reason about.
SUPPORTED_CURRENCIES: frozenset[Currency] = frozenset({Currency.INR, Currency.USD})

#: Total stacked discount above this share of the listed price is rejected as
#: incoherent data rather than applied.
MAX_TOTAL_DISCOUNT_RATIO = Decimal("0.90")


class PriceValidationReport(BaseModel):
    """Why a price was accepted or rejected. Recorded in the audit trail."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PriceValidator:
    """Validates price data and computes the effective price."""

    def validate_product_price(self, product: Product) -> PriceValidationReport:
        """Check the product's own price fields before any offer is considered."""
        errors: list[str] = []
        warnings: list[str] = []

        listed = product.listed_price
        if listed.currency not in SUPPORTED_CURRENCIES:
            errors.append(f"unsupported_currency:{listed.currency}")
        if listed.amount <= 0:
            errors.append("listed_price_not_positive")

        if product.mrp is not None:
            if product.mrp.currency is not listed.currency:
                errors.append("currency_mismatch_listed_vs_mrp")
            elif listed > product.mrp:
                errors.append("listed_price_exceeds_mrp")
            elif product.mrp.amount > 0:
                implied = (product.mrp.amount - listed.amount) / product.mrp.amount
                if implied > MAX_TOTAL_DISCOUNT_RATIO:
                    errors.append("implausible_mrp_discount")

        return PriceValidationReport(
            is_valid=not errors, errors=errors, warnings=warnings
        )

    def compute(
        self,
        product: Product,
        offers: list[Offer],
        profile: PurchaseProfile,
    ) -> tuple[PriceBreakdown, PriceValidationReport]:
        """Compute the effective price for ``product`` under ``profile``.

        Returns the breakdown together with the report. When the report is
        invalid, the breakdown carries ``is_valid=False`` and a zero discount —
        an unreliable price is never turned into an attractive one.
        """
        listed = product.listed_price
        base_report = self.validate_product_price(product)
        errors = list(base_report.errors)
        warnings = list(base_report.warnings)

        if errors:
            return self._invalid_breakdown(listed, errors, warnings), PriceValidationReport(
                is_valid=False, errors=errors, warnings=warnings
            )

        currency = listed.currency
        applied: list[AppliedOffer] = []
        unmet: list[str] = []
        cashback = Money.zero(currency)
        running_discount = Money.zero(currency)

        seen_offer_ids: set[str] = set()
        #: Guards against two distinct records expressing the same discount.
        seen_signatures: set[tuple[str, str]] = set()

        for offer in self._deterministic_order(offers):
            if offer.product_id != product.product_id or offer.marketplace is not product.marketplace:
                errors.append(f"offer_product_mismatch:{offer.offer_id}")
                continue
            if offer.value.currency is not currency:
                errors.append(f"offer_currency_mismatch:{offer.offer_id}")
                continue
            if offer.value.amount < 0:
                errors.append(f"negative_discount:{offer.offer_id}")
                continue

            if offer.offer_id in seen_offer_ids:
                warnings.append(f"duplicate_offer_id_ignored:{offer.offer_id}")
                continue
            seen_offer_ids.add(offer.offer_id)

            signature = (offer.kind.value, str(offer.value.amount))
            if signature in seen_signatures:
                # Same kind, same amount, different id: almost certainly the same
                # promotion listed twice. Applying both would double the discount.
                warnings.append(f"duplicate_discount_ignored:{offer.offer_id}")
                continue

            # ---- cashback is never an upfront reduction ----
            if offer.kind is OfferKind.CASHBACK:
                if offer.value > listed:
                    errors.append(f"cashback_exceeds_listed_price:{offer.offer_id}")
                    continue
                cashback = cashback + offer.value
                seen_signatures.add(signature)
                continue

            # ---- financing terms are not a price change ----
            if offer.kind is OfferKind.NO_COST_EMI:
                if offer.value.amount != 0:
                    errors.append(f"no_cost_emi_with_nonzero_value:{offer.offer_id}")
                continue

            # ---- conditional offers are applied only when actually eligible ----
            if not offer.applies_to(profile, listed):
                if offer.kind.is_conditional:
                    unmet.append(offer.offer_id)
                continue

            if offer.value > listed:
                errors.append(f"discount_exceeds_listed_price:{offer.offer_id}")
                continue

            candidate_total = running_discount + offer.value
            if candidate_total > listed:
                # Stacking would push the price below zero.
                errors.append(f"stacked_discount_exceeds_listed_price:{offer.offer_id}")
                continue
            if listed.amount > 0 and (
                candidate_total.amount / listed.amount > MAX_TOTAL_DISCOUNT_RATIO
            ):
                errors.append(f"implausible_total_discount:{offer.offer_id}")
                continue
            if not offer.stackable and applied:
                warnings.append(f"non_stackable_offer_skipped:{offer.offer_id}")
                continue

            running_discount = candidate_total
            seen_signatures.add(signature)
            applied.append(
                AppliedOffer(
                    offer_id=offer.offer_id,
                    kind=offer.kind,
                    amount=offer.value,
                    reason=self._reason_for(offer),
                )
            )

        if errors:
            return self._invalid_breakdown(listed, errors, warnings), PriceValidationReport(
                is_valid=False, errors=errors, warnings=warnings
            )

        effective = listed - running_discount
        if effective.amount < 0:  # pragma: no cover - clamped by Money.__sub__
            errors.append("effective_price_negative")
            return self._invalid_breakdown(listed, errors, warnings), PriceValidationReport(
                is_valid=False, errors=errors, warnings=warnings
            )

        if unmet:
            warnings.append(f"conditional_offers_not_applied:{len(unmet)}")
        if cashback.amount > 0:
            warnings.append("cashback_excluded_from_effective_price")

        breakdown = PriceBreakdown(
            listed_price=listed,
            applied_offers=applied,
            total_upfront_discount=running_discount,
            effective_price=effective,
            cashback_value=cashback,
            unmet_conditional_offers=sorted(unmet),
            warnings=warnings,
            is_valid=True,
        )
        return breakdown, PriceValidationReport(is_valid=True, warnings=warnings)

    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_order(offers: list[Offer]) -> list[Offer]:
        """Sort offers so the computation is reproducible.

        Reproducibility matters: the recommendation validator recomputes the
        price and compares. Iteration order must not depend on how the provider
        happened to order its response. Largest unconditional discounts first,
        then by id for a total order.
        """
        return sorted(
            offers,
            key=lambda offer: (
                offer.kind.is_conditional,
                -offer.value.amount,
                offer.offer_id,
            ),
        )

    @staticmethod
    def _reason_for(offer: Offer) -> str:
        if offer.kind is OfferKind.BANK_DISCOUNT and offer.requires_bank:
            return f"applied: requires {offer.requires_bank.value.upper()} card"
        if offer.kind is OfferKind.EXCHANGE_BONUS:
            return "applied: requires device exchange"
        if offer.kind is OfferKind.COUPON:
            return "applied: coupon"
        return "applied: upfront discount"

    @staticmethod
    def _invalid_breakdown(
        listed: Money, errors: list[str], warnings: list[str]
    ) -> PriceBreakdown:
        """A breakdown for unusable price data: no discount, flagged invalid."""
        return PriceBreakdown(
            listed_price=listed,
            applied_offers=[],
            total_upfront_discount=Money.zero(listed.currency),
            effective_price=listed,
            cashback_value=Money.zero(listed.currency),
            unmet_conditional_offers=[],
            warnings=[*warnings, *errors],
            is_valid=False,
        )
