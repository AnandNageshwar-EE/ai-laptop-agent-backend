"""Marketplace response validation.

This is the boundary where raw provider JSON stops being raw. A payload that
fails any check is **quarantined with a reason** rather than dropped silently —
a marketplace quietly returning malformed prices should be visible in the audit
trail, not invisible because the products vanished.

Checks applied here, before anything reaches graph state:

* the envelope has the expected shape
* every required identifier is present and well-formed
* the URL parses, uses https, and its host belongs to the claimed marketplace
* prices are positive, coherent (``listed <= mrp``) and in a supported currency
* discounts are non-negative and not larger than the listed price
* the marketplace field matches the provider that actually answered
* injection attempts in seller-authored text are recorded, and the text is kept
  as data rather than causing the product to be rejected
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..domain.enums import Marketplace, OfferKind, ProductCategory
from ..domain.money import Money, MoneyInput
from ..domain.product import LaptopSpecs, Offer, Product
from .injection import scan_for_injection
from .result import ValidationOutcome

#: Hosts a product URL may legitimately live on, per marketplace. A URL outside
#: this set is rejected even if the provider claims it — it is the only defence
#: against a compromised or spoofed provider steering users off-platform.
TRUSTED_HOSTS: dict[Marketplace, frozenset[str]] = {
    Marketplace.AMAZON: frozenset(
        {"www.amazon.in", "amazon.in", "www.amazon.com", "amazon.com"}
    ),
    Marketplace.FLIPKART: frozenset({"www.flipkart.com", "flipkart.com"}),
}

#: A discount larger than this share of the listed price is treated as
#: impossible data rather than a genuine offer.
MAX_PLAUSIBLE_DISCOUNT_RATIO = 0.90


class ProductPayload(BaseModel):
    """Untrusted product envelope exactly as a provider returns it.

    Permissive about *types* (so a malformed field yields a quarantine record
    rather than an exception) and strict about *presence*.
    """

    model_config = ConfigDict(extra="ignore")

    product_id: str | None = None
    title: str | None = None
    brand: str | None = None
    url: str | None = None
    price: MoneyInput | None = None
    mrp: MoneyInput | None = None
    rating: float | None = None
    rating_count: int | None = None
    in_stock: bool = True
    description: str = ""
    specs: dict[str, Any] = Field(default_factory=dict)


class OfferPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offer_id: str | None = None
    product_id: str | None = None
    kind: str | None = None
    value: MoneyInput | None = None
    percent: float | None = None
    max_discount: MoneyInput | None = None
    description: str = ""
    requires_bank: str | None = None
    requires_exchange: bool = False
    min_transaction: MoneyInput | None = None
    stackable: bool = True


class MarketplaceResponseValidator:
    """Turns untrusted provider payloads into validated domain objects."""

    def __init__(self, marketplace: Marketplace) -> None:
        self._marketplace = marketplace
        self._trusted_hosts = TRUSTED_HOSTS[marketplace]
        #: Injection categories found in seller text, for the audit trail.
        self.flagged_content: list[tuple[str, list[str]]] = []

    # ------------------------------------------------------------------
    # products
    # ------------------------------------------------------------------

    def validate_products(self, raw: Any) -> ValidationOutcome[Product]:
        outcome: ValidationOutcome[Product] = ValidationOutcome()

        items = self._unwrap_list(raw, key="products")
        if items is None:
            outcome.quarantined.append(("<envelope>", "unexpected_response_structure"))
            return outcome

        seen_ids: set[str] = set()
        for index, item in enumerate(items):
            identifier = f"index:{index}"
            try:
                payload = ProductPayload.model_validate(item)
            except ValidationError as exc:
                outcome.quarantined.append((identifier, f"malformed_payload:{_first_error(exc)}"))
                continue

            identifier = payload.product_id or identifier
            problem = self._check_product(payload)
            if problem is not None:
                outcome.quarantined.append((identifier, problem))
                continue

            assert payload.product_id is not None  # guaranteed by _check_product
            if payload.product_id in seen_ids:
                outcome.quarantined.append((identifier, "duplicate_product_id"))
                continue

            try:
                product = self._build_product(payload)
            except (ValidationError, ValueError) as exc:
                reason = _first_error(exc) if isinstance(exc, ValidationError) else str(exc)
                outcome.quarantined.append((identifier, f"failed_domain_validation:{reason}"))
                continue

            self._record_content_flags(product.product_id, payload)
            seen_ids.add(product.product_id)
            outcome.accepted.append(product)

        return outcome

    def _check_product(self, payload: ProductPayload) -> str | None:
        """Return a quarantine reason, or ``None`` if the payload is acceptable."""
        if not payload.product_id or not payload.product_id.strip():
            return "missing_product_id"
        if not payload.title or not payload.title.strip():
            return "missing_title"
        if payload.price is None:
            return "missing_price"

        try:
            price = payload.price.to_money()
        except (ValidationError, ValueError) as exc:
            return f"invalid_price:{_short(exc)}"
        if price.amount <= 0:
            return "non_positive_price"

        if payload.mrp is not None:
            try:
                mrp = payload.mrp.to_money()
            except (ValidationError, ValueError) as exc:
                return f"invalid_mrp:{_short(exc)}"
            if mrp.currency is not price.currency:
                return "currency_mismatch_price_vs_mrp"
            if price > mrp:
                return "listed_price_exceeds_mrp"
            # An "MRP" inflated far above the selling price fakes a discount.
            if mrp.amount > 0 and price.amount / mrp.amount < (
                1 - MAX_PLAUSIBLE_DISCOUNT_RATIO
            ):
                return "implausible_mrp_discount"

        url_problem = self._check_url(payload.url)
        if url_problem is not None:
            return url_problem

        if not payload.specs:
            return "missing_specs"
        return None

    def _check_url(self, url: str | None) -> str | None:
        if not url:
            return "missing_url"
        try:
            parsed = urlparse(url)
        except ValueError:
            return "malformed_url"
        if parsed.scheme != "https":
            return f"disallowed_url_scheme:{parsed.scheme or 'none'}"
        host = (parsed.hostname or "").lower()
        if not host:
            return "malformed_url"
        if host not in self._trusted_hosts:
            # The single most important check here: a provider cannot hand us a
            # link to somewhere else and have it surfaced as a recommendation.
            return f"untrusted_url_host:{host}"
        if parsed.username or parsed.password:
            return "url_contains_credentials"
        return None

    def _build_product(self, payload: ProductPayload) -> Product:
        assert payload.product_id and payload.title and payload.price
        return Product(
            product_id=payload.product_id.strip(),
            marketplace=self._marketplace,
            category=ProductCategory.LAPTOP,
            title=payload.title.strip(),
            brand=(payload.brand or _brand_from_title(payload.title)).strip(),
            url=payload.url,  # type: ignore[arg-type]
            listed_price=payload.price.to_money(),
            mrp=payload.mrp.to_money() if payload.mrp else None,
            rating=payload.rating,
            rating_count=payload.rating_count or 0,
            in_stock=payload.in_stock,
            specs=LaptopSpecs.model_validate(payload.specs),
            description=payload.description[:2000],
        )

    def _record_content_flags(self, product_id: str, payload: ProductPayload) -> None:
        """Flag seller text that tries to instruct the model.

        The product is deliberately *not* rejected: a competitor could otherwise
        remove a rival listing from the user's results by poisoning its
        description. The text stays as data and the attempt is recorded.
        """
        combined = f"{payload.title or ''}\n{payload.description}"
        scan = scan_for_injection(combined, check_structure=False)
        if scan.detected:
            self.flagged_content.append((product_id, scan.categories))

    # ------------------------------------------------------------------
    # offers
    # ------------------------------------------------------------------

    def validate_offers(
        self, raw: Any, *, known_products: dict[str, Money]
    ) -> ValidationOutcome[Offer]:
        """Validate offers against the products actually retrieved.

        ``known_products`` maps product id to listed price. An offer for a
        product we did not retrieve is quarantined — it has nothing to apply to,
        and accepting it would let a provider attach a discount to an arbitrary id.
        """
        outcome: ValidationOutcome[Offer] = ValidationOutcome()

        items = self._unwrap_list(raw, key="offers")
        if items is None:
            outcome.quarantined.append(("<envelope>", "unexpected_response_structure"))
            return outcome

        seen: set[str] = set()
        for index, item in enumerate(items):
            identifier = f"index:{index}"
            try:
                payload = OfferPayload.model_validate(item)
            except ValidationError as exc:
                outcome.quarantined.append((identifier, f"malformed_payload:{_first_error(exc)}"))
                continue

            identifier = payload.offer_id or identifier
            if not payload.offer_id or not payload.product_id:
                outcome.quarantined.append((identifier, "missing_identifier"))
                continue
            if payload.product_id not in known_products:
                outcome.quarantined.append((identifier, "offer_for_unknown_product"))
                continue
            if payload.offer_id in seen:
                # Duplicate offer ids are how a double discount gets applied.
                outcome.quarantined.append((identifier, "duplicate_offer_id"))
                continue
            if payload.kind not in {kind.value for kind in OfferKind}:
                outcome.quarantined.append((identifier, f"unknown_offer_kind:{payload.kind}"))
                continue

            listed = known_products[payload.product_id]
            resolved = self._resolve_offer_value(payload, listed)
            if isinstance(resolved, str):
                outcome.quarantined.append((identifier, resolved))
                continue

            try:
                offer = Offer(
                    offer_id=payload.offer_id,
                    marketplace=self._marketplace,
                    product_id=payload.product_id,
                    kind=OfferKind(payload.kind),
                    value=resolved,
                    description=payload.description[:500],
                    requires_bank=payload.requires_bank,  # type: ignore[arg-type]
                    requires_exchange=payload.requires_exchange,
                    min_transaction=(
                        payload.min_transaction.to_money() if payload.min_transaction else None
                    ),
                    stackable=payload.stackable,
                )
            except (ValidationError, ValueError) as exc:
                reason = _first_error(exc) if isinstance(exc, ValidationError) else str(exc)
                outcome.quarantined.append((identifier, f"failed_domain_validation:{reason}"))
                continue

            seen.add(offer.offer_id)
            outcome.accepted.append(offer)

        return outcome

    def _resolve_offer_value(
        self, payload: OfferPayload, listed: Money
    ) -> Money | str:
        """Resolve an offer to an absolute amount, or return a quarantine reason."""
        kind = OfferKind(payload.kind) if payload.kind else None

        if kind is OfferKind.NO_COST_EMI:
            return Money.zero(listed.currency)

        if payload.value is not None:
            try:
                value = payload.value.to_money()
            except (ValidationError, ValueError) as exc:
                return f"invalid_offer_value:{_short(exc)}"
        elif payload.percent is not None:
            if not (0 < payload.percent <= 100):
                return f"invalid_offer_percent:{payload.percent}"
            value = Money(
                amount=listed.amount * (Decimal(str(payload.percent)) / Decimal("100")),
                currency=listed.currency,
            )
        else:
            return "offer_has_no_value"

        if value.currency is not listed.currency:
            return "offer_currency_mismatch"
        if value.amount < 0:
            return "negative_offer_value"

        # Cap a percentage offer at its stated ceiling.
        if payload.max_discount is not None:
            try:
                cap = payload.max_discount.to_money()
            except (ValidationError, ValueError) as exc:
                return f"invalid_max_discount:{_short(exc)}"
            if cap.currency is not listed.currency:
                return "max_discount_currency_mismatch"
            if value > cap:
                value = cap

        if kind is not None and kind.reduces_upfront_price:
            if value > listed:
                return "discount_exceeds_listed_price"
            if listed.amount > 0 and value.amount / listed.amount > Decimal(
                str(MAX_PLAUSIBLE_DISCOUNT_RATIO)
            ):
                return "implausible_discount_ratio"

        return value

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_list(raw: Any, *, key: str) -> list[Any] | None:
        """Accept either a bare list or ``{key: [...]}``; reject anything else."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        return None


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "unknown"
    error = errors[0]
    location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
    return f"{location}:{error.get('type', 'invalid')}"


def _short(exc: Exception) -> str:
    return str(exc).splitlines()[0][:80]


def _brand_from_title(title: str) -> str:
    """Fallback brand extraction. First token only, never free-form parsing."""
    token = title.strip().split(" ", 1)[0]
    return token[:32] or "unknown"
