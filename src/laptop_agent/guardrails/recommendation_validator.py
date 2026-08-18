"""Final recommendation validation.

This is the last gate before an answer is returned, and it is deliberately
independent of the code that produced the recommendation: it re-derives the
price, re-evaluates the hard constraints and re-computes the score from the
provider data, then compares. A bug or a manipulation upstream shows up as a
mismatch here rather than as a wrong answer to the user.

Checks, per spec section 1.7:

* the candidate exists
* the candidate was actually returned by a marketplace provider in *this* run
* the product URL is byte-identical to the provider's URL, on a trusted host
* hard requirements pass
* the price calculation is valid and reproducible
* the score is reproducible
* no mandatory constraint is violated
* the recommendation belongs to the ranked candidate set

On failure the graph routes back and re-ranks with the offending candidate
excluded. The LLM is never consulted about the verdict and cannot overturn it.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import Marketplace
from ..domain.product import Product, ProductCandidate, evaluate_hard_constraints
from ..domain.recommendation import Recommendation
from ..domain.requirements import LaptopRequirements, PurchaseProfile
from ..domain.scoring import SCORE_EPSILON, ProductScore
from .price_validator import PriceValidator
from .tool_output import TRUSTED_HOSTS

#: Signature of the scoring function, injected so the validator recomputes the
#: score with the same code the ranking node used — without importing ranking
#: and creating a cycle.
ScoreFn = Callable[[ProductCandidate, LaptopRequirements], ProductScore]


class ValidationFailure(StrEnum):
    NO_RECOMMENDATION = "no_recommendation"
    CANDIDATE_NOT_FOUND = "candidate_not_found"
    NOT_FROM_PROVIDER = "not_from_provider"
    URL_NOT_FROM_PROVIDER = "url_not_from_provider"
    URL_UNTRUSTED_HOST = "url_untrusted_host"
    HARD_REQUIREMENT_FAILED = "hard_requirement_failed"
    MANDATORY_CONSTRAINT_VIOLATED = "mandatory_constraint_violated"
    PRICE_INVALID = "price_invalid"
    PRICE_NOT_REPRODUCIBLE = "price_not_reproducible"
    SCORE_NOT_REPRODUCIBLE = "score_not_reproducible"
    NOT_IN_RANKED_SET = "not_in_ranked_set"
    OUT_OF_STOCK = "out_of_stock"
    FIELD_MISMATCH = "field_mismatch"
    SELLER_CONTENT_FLAGGED = "seller_content_flagged"


class RecommendationValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    failures: list[ValidationFailure] = Field(default_factory=list)
    #: Machine-readable detail for the audit trail.
    detail: dict[str, str] = Field(default_factory=dict)

    @property
    def rejected_reason(self) -> str:
        return ",".join(failure.value for failure in self.failures)


class ProviderRegistry:
    """Products a provider actually returned during this run.

    Built only from :class:`~laptop_agent.guardrails.tool_output.MarketplaceResponseValidator`
    output. Because it is constructed in-process per run and never rehydrated
    from a client or a cache, membership here is a genuine provenance proof —
    which is precisely why product data is not carried across turns.
    """

    def __init__(self) -> None:
        self._products: dict[tuple[Marketplace, str], Product] = {}

    def register_all(self, products: list[Product]) -> None:
        for product in products:
            self._products[product.key] = product

    def get(self, key: tuple[Marketplace, str]) -> Product | None:
        return self._products.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._products

    def __len__(self) -> int:
        return len(self._products)

    @property
    def keys(self) -> list[tuple[Marketplace, str]]:
        return list(self._products)


class RecommendationValidator:
    """Independently re-derives and verifies a recommendation."""

    def __init__(self, price_validator: PriceValidator | None = None) -> None:
        self._prices = price_validator or PriceValidator()

    def validate(
        self,
        *,
        recommendation: Recommendation | None,
        candidates: list[ProductCandidate],
        ranked: list[ProductScore],
        requirements: LaptopRequirements,
        profile: PurchaseProfile,
        registry: ProviderRegistry,
        rescore: ScoreFn,
    ) -> RecommendationValidationResult:
        failures: list[ValidationFailure] = []
        detail: dict[str, str] = {}

        if recommendation is None:
            return RecommendationValidationResult(
                is_valid=False, failures=[ValidationFailure.NO_RECOMMENDATION]
            )

        key = recommendation.key
        detail["candidate"] = f"{key[0].value}:{key[1]}"

        # ---- 1. the candidate exists in the candidate set ----
        candidate = next((c for c in candidates if c.key == key), None)
        if candidate is None:
            return RecommendationValidationResult(
                is_valid=False,
                failures=[ValidationFailure.CANDIDATE_NOT_FOUND],
                detail=detail,
            )

        # ---- 2. it came from a provider in this run ----
        provider_product = registry.get(key)
        if provider_product is None:
            return RecommendationValidationResult(
                is_valid=False,
                failures=[ValidationFailure.NOT_FROM_PROVIDER],
                detail=detail,
            )

        # ---- 3. URL provenance ----
        recommended_url = str(recommendation.url)
        provider_url = str(provider_product.url)
        if recommended_url != provider_url:
            failures.append(ValidationFailure.URL_NOT_FROM_PROVIDER)
            detail["url"] = "differs_from_provider_url"
        host = (urlparse(recommended_url).hostname or "").lower()
        if host not in TRUSTED_HOSTS.get(recommendation.marketplace, frozenset()):
            failures.append(ValidationFailure.URL_UNTRUSTED_HOST)
            detail["url_host"] = host or "none"

        # ---- 4. the recommendation's own fields match the provider product ----
        if recommendation.title != provider_product.title:
            failures.append(ValidationFailure.FIELD_MISMATCH)
            detail["title"] = "differs_from_provider_title"
        if recommendation.specs != provider_product.specs:
            # Specifications drive the hard-constraint checks the user relies on,
            # so a mismatch here is as serious as a wrong price.
            failures.append(ValidationFailure.FIELD_MISMATCH)
            detail["specs"] = "differs_from_provider_specs"
        if not provider_product.in_stock:
            failures.append(ValidationFailure.OUT_OF_STOCK)

        # ---- 5. trust gate: a manipulation attempt disqualifies the listing ----
        if candidate.trust_flagged:
            failures.append(ValidationFailure.SELLER_CONTENT_FLAGGED)
            detail["trust"] = "seller_text_attempted_manipulation"

        # ---- 6. hard requirements, re-evaluated from provider data ----
        if not candidate.hard_requirements_passed:
            failures.append(ValidationFailure.HARD_REQUIREMENT_FAILED)
            detail["failed_constraints"] = ",".join(candidate.failed_constraints)

        # ---- 6. price: valid, and reproducible from provider data ----
        recomputed_price, price_report = self._prices.compute(
            provider_product, candidate.offers, profile
        )
        if not price_report.is_valid:
            failures.append(ValidationFailure.PRICE_INVALID)
            detail["price_errors"] = ",".join(price_report.errors)[:200]
        elif (
            recomputed_price.effective_price.amount
            != recommendation.effective_price.amount
            or recomputed_price.effective_price.currency
            is not recommendation.effective_price.currency
            or recomputed_price.listed_price.amount != recommendation.listed_price.amount
            or recomputed_price.total_upfront_discount.amount
            != recommendation.upfront_savings.amount
            or recomputed_price.cashback_value.amount != recommendation.cashback_value.amount
        ):
            failures.append(ValidationFailure.PRICE_NOT_REPRODUCIBLE)
            detail["price"] = (
                f"expected_effective={recomputed_price.effective_price.amount}"
                f" got={recommendation.effective_price.amount}"
            )

        # ---- 7. mandatory constraints, re-derived against the recomputed price ----
        if price_report.is_valid:
            violated = evaluate_hard_constraints(
                provider_product, requirements, recomputed_price
            )
            if violated:
                failures.append(ValidationFailure.MANDATORY_CONSTRAINT_VIOLATED)
                detail["violated"] = ",".join(violated)

        # ---- 8. the score is reproducible ----
        stored = next((score for score in ranked if score.key == key), None)
        if stored is None:
            failures.append(ValidationFailure.NOT_IN_RANKED_SET)
        else:
            if abs(stored.total - recommendation.score) > SCORE_EPSILON:
                failures.append(ValidationFailure.SCORE_NOT_REPRODUCIBLE)
                detail["score"] = f"ranked={stored.total} recommended={recommendation.score}"
            elif price_report.is_valid:
                recomputed_score = rescore(
                    candidate.model_copy(update={"price": recomputed_price}), requirements
                )
                if not recomputed_score.matches(stored):
                    failures.append(ValidationFailure.SCORE_NOT_REPRODUCIBLE)
                    detail["score_recomputed"] = (
                        f"stored={stored.total} recomputed={recomputed_score.total}"
                    )

        # ---- 9. the recommendation must be the top of the ranked set ----
        if ranked and not failures:
            best = max(ranked, key=lambda score: (score.total, score.product_id))
            if best.key != key and abs(best.total - recommendation.score) > SCORE_EPSILON:
                failures.append(ValidationFailure.NOT_IN_RANKED_SET)
                detail["ranking"] = "not_the_highest_scoring_candidate"

        return RecommendationValidationResult(
            is_valid=not failures, failures=failures, detail=detail
        )
