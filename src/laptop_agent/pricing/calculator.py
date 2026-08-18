"""Candidate assembly: product + offers -> priced, constraint-checked candidate.

Deterministic and LLM-free. Given the same provider data, requirements and
profile, this produces byte-identical output every time — which is what lets the
recommendation validator re-run it and compare.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import Marketplace
from ..domain.product import (
    Offer,
    Product,
    ProductCandidate,
    evaluate_hard_constraints,
)
from ..domain.requirements import LaptopRequirements, PurchaseProfile
from ..guardrails.price_validator import PriceValidator


class CandidateBuildReport(BaseModel):
    """What happened while assembling candidates, for the audit trail."""

    model_config = ConfigDict(extra="forbid")

    built: int = 0
    #: (product key, reason) for products excluded before ranking.
    rejected: list[tuple[str, str]] = Field(default_factory=list)
    price_errors: list[tuple[str, str]] = Field(default_factory=list)
    #: Listings disqualified because their seller text attempted manipulation.
    trust_flagged: list[str] = Field(default_factory=list)


class PriceCalculator:
    """Builds :class:`ProductCandidate` objects from validated provider data."""

    def __init__(self, price_validator: PriceValidator | None = None) -> None:
        self._prices = price_validator or PriceValidator()

    def build_candidates(
        self,
        products: list[Product],
        offers: list[Offer],
        requirements: LaptopRequirements,
        profile: PurchaseProfile,
        flagged_product_ids: set[str] | None = None,
    ) -> tuple[list[ProductCandidate], CandidateBuildReport]:
        report = CandidateBuildReport()
        flagged = flagged_product_ids or set()
        offers_by_product = self._index_offers(offers)
        candidates: list[ProductCandidate] = []

        for product in sorted(products, key=lambda p: (p.marketplace.value, p.product_id)):
            key = f"{product.marketplace.value}:{product.product_id}"
            product_offers = offers_by_product.get(product.key, [])

            price, price_report = self._prices.compute(product, product_offers, profile)
            if not price_report.is_valid:
                # Inconsistent price data: the product is not recommendable.
                report.rejected.append((key, "price_invalid"))
                report.price_errors.append((key, ",".join(price_report.errors)[:200]))
                continue

            if not product.in_stock:
                report.rejected.append((key, "out_of_stock"))
                continue

            # A currency the user did not ask in cannot be compared to a budget.
            if product.currency is not requirements.currency:
                report.rejected.append((key, "currency_mismatch_with_requirements"))
                continue

            failed = evaluate_hard_constraints(product, requirements, price)
            is_flagged = product.product_id in flagged
            if is_flagged:
                report.trust_flagged.append(key)
            candidates.append(
                ProductCandidate(
                    product=product,
                    offers=sorted(product_offers, key=lambda offer: offer.offer_id),
                    price=price,
                    hard_requirements_passed=not failed,
                    failed_constraints=failed,
                    trust_flagged=is_flagged,
                )
            )
            report.built += 1

        return candidates, report

    @staticmethod
    def _index_offers(
        offers: list[Offer],
    ) -> dict[tuple[Marketplace, str], list[Offer]]:
        index: dict[tuple[Marketplace, str], list[Offer]] = {}
        for offer in offers:
            index.setdefault(offer.key, []).append(offer)
        return index
