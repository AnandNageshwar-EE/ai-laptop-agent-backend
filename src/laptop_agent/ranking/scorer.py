"""Deterministic, reproducible scoring.

Reproducibility is a hard requirement: the recommendation validator recomputes
the winner's score and rejects the recommendation if it does not match. So:

* every component is a pure function of the candidate and the requirements
* no randomness, no clock, no dict iteration order, no floats — ``Decimal`` only
* weights are constants keyed by use case, and they sum to exactly 1
* ties break on a stable key, never on list order

A change to weights or component maths requires bumping
:data:`~laptop_agent.domain.scoring.SCORING_VERSION`, so a stored score is never
compared against a score computed by different code.
"""

from __future__ import annotations

from decimal import Decimal

from ..domain.enums import UseCase
from ..domain.product import ProductCandidate
from ..domain.requirements import LaptopRequirements
from ..domain.scoring import SCORING_VERSION, ProductScore, ScoreComponent

_ONE = Decimal("1")
_ZERO = Decimal("0")

#: Component weights per use case. Each row sums to exactly 1.
_WEIGHTS: dict[UseCase, dict[str, Decimal]] = {
    UseCase.GENERAL: {
        "requirement_fit": Decimal("0.30"),
        "value_for_money": Decimal("0.25"),
        "budget_headroom": Decimal("0.15"),
        "spec_strength": Decimal("0.20"),
        "rating_confidence": Decimal("0.10"),
    },
    UseCase.STUDENT: {
        "requirement_fit": Decimal("0.30"),
        "value_for_money": Decimal("0.30"),
        "budget_headroom": Decimal("0.20"),
        "spec_strength": Decimal("0.10"),
        "rating_confidence": Decimal("0.10"),
    },
    UseCase.OFFICE_PRODUCTIVITY: {
        "requirement_fit": Decimal("0.35"),
        "value_for_money": Decimal("0.25"),
        "budget_headroom": Decimal("0.15"),
        "spec_strength": Decimal("0.15"),
        "rating_confidence": Decimal("0.10"),
    },
    UseCase.SOFTWARE_DEVELOPMENT: {
        "requirement_fit": Decimal("0.35"),
        "value_for_money": Decimal("0.20"),
        "budget_headroom": Decimal("0.10"),
        "spec_strength": Decimal("0.25"),
        "rating_confidence": Decimal("0.10"),
    },
    UseCase.DATA_SCIENCE: {
        "requirement_fit": Decimal("0.30"),
        "value_for_money": Decimal("0.15"),
        "budget_headroom": Decimal("0.10"),
        "spec_strength": Decimal("0.35"),
        "rating_confidence": Decimal("0.10"),
    },
    UseCase.GAMING: {
        "requirement_fit": Decimal("0.30"),
        "value_for_money": Decimal("0.20"),
        "budget_headroom": Decimal("0.10"),
        "spec_strength": Decimal("0.35"),
        "rating_confidence": Decimal("0.05"),
    },
    UseCase.CONTENT_CREATION: {
        "requirement_fit": Decimal("0.30"),
        "value_for_money": Decimal("0.15"),
        "budget_headroom": Decimal("0.10"),
        "spec_strength": Decimal("0.35"),
        "rating_confidence": Decimal("0.10"),
    },
}

#: Reference ceilings used to normalise raw specs into 0..1.
_RAM_CEILING = Decimal("32")
_STORAGE_CEILING = Decimal("1024")
_BATTERY_CEILING = Decimal("18")

#: What "spec strength" means per use case: (attribute, weight) pairs summing to 1.
_SPEC_PROFILE: dict[UseCase, tuple[tuple[str, Decimal], ...]] = {
    UseCase.GENERAL: (("ram", Decimal("0.4")), ("storage", Decimal("0.3")), ("battery", Decimal("0.3"))),
    UseCase.STUDENT: (("battery", Decimal("0.4")), ("portability", Decimal("0.3")), ("ram", Decimal("0.3"))),
    UseCase.OFFICE_PRODUCTIVITY: (("battery", Decimal("0.4")), ("portability", Decimal("0.3")), ("ram", Decimal("0.3"))),
    UseCase.SOFTWARE_DEVELOPMENT: (("ram", Decimal("0.5")), ("storage", Decimal("0.3")), ("battery", Decimal("0.2"))),
    UseCase.DATA_SCIENCE: (("ram", Decimal("0.4")), ("gpu", Decimal("0.4")), ("storage", Decimal("0.2"))),
    UseCase.GAMING: (("gpu", Decimal("0.5")), ("refresh", Decimal("0.3")), ("ram", Decimal("0.2"))),
    UseCase.CONTENT_CREATION: (("gpu", Decimal("0.4")), ("ram", Decimal("0.4")), ("storage", Decimal("0.2"))),
}


def _clamp(value: Decimal) -> Decimal:
    """Constrain to 0..1 and fix the precision so results are bit-stable."""
    bounded = max(_ZERO, min(_ONE, value))
    return bounded.quantize(Decimal("0.000001"))


def _ratio(value: Decimal, ceiling: Decimal) -> Decimal:
    return _clamp(value / ceiling) if ceiling > 0 else _ZERO


def _dec(value: float | int | None) -> Decimal | None:
    """Decimal, or ``None`` when the specification is unknown."""
    return None if value is None else Decimal(str(value))


def _credit(value: float | int | None, ceiling: Decimal) -> Decimal:
    """Normalised credit for a spec, or zero when it is unknown.

    Unknown earns no credit rather than an average. A listing that does not
    report a specification must not outrank one that does and reports it poorly —
    that would reward incomplete data.
    """
    number = _dec(value)
    return _ZERO if number is None else _ratio(number, ceiling)


def score_candidate(
    candidate: ProductCandidate, requirements: LaptopRequirements
) -> ProductScore:
    """Score one candidate. Pure function — same inputs, same output, always."""
    weights = _WEIGHTS[requirements.use_case]
    components = [
        ScoreComponent(
            name="requirement_fit",
            weight=weights["requirement_fit"],
            raw=_requirement_fit(candidate, requirements),
        ),
        ScoreComponent(
            name="value_for_money",
            weight=weights["value_for_money"],
            raw=_value_for_money(candidate, requirements),
        ),
        ScoreComponent(
            name="budget_headroom",
            weight=weights["budget_headroom"],
            raw=_budget_headroom(candidate, requirements),
        ),
        ScoreComponent(
            name="spec_strength",
            weight=weights["spec_strength"],
            raw=_spec_strength(candidate, requirements),
        ),
        ScoreComponent(
            name="rating_confidence",
            weight=weights["rating_confidence"],
            raw=_rating_confidence(candidate),
        ),
    ]
    total = sum((component.weighted for component in components), start=_ZERO)
    return ProductScore(
        product_id=candidate.product.product_id,
        marketplace=candidate.product.marketplace,
        components=components,
        total=total.quantize(Decimal("0.000001")),
        scoring_version=SCORING_VERSION,
    )


def rank_candidates(
    candidates: list[ProductCandidate],
    requirements: LaptopRequirements,
    *,
    exclude: set[tuple[str, str]] | None = None,
) -> list[ProductScore]:
    """Score and order the recommendable candidates, best first.

    ``exclude`` holds ``(marketplace, product_id)`` pairs rejected by the
    recommendation validator on a previous attempt, so a re-rank cannot select
    the same failing candidate again.
    """
    excluded = exclude or set()
    scores = [
        score_candidate(candidate, requirements)
        for candidate in candidates
        if candidate.is_recommendable
        and (candidate.product.marketplace.value, candidate.product.product_id)
        not in excluded
    ]
    # Stable total order: score desc, then marketplace and id ascending.
    return sorted(
        scores,
        key=lambda score: (-score.total, score.marketplace.value, score.product_id),
    )


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


def _requirement_fit(
    candidate: ProductCandidate, requirements: LaptopRequirements
) -> Decimal:
    """Share of the *stated* preferences this candidate satisfies.

    Mandatory constraints are not scored here — a candidate that violates one is
    removed before ranking, so scoring them again would double-count.
    """
    specs = candidate.product.specs
    checks: list[bool] = []

    # An unknown specification counts as unsatisfied: it is not evidence of fit.
    if requirements.min_ram_gb is not None:
        checks.append(specs.ram_gb is not None and specs.ram_gb >= requirements.min_ram_gb)
    if requirements.min_storage_gb is not None:
        checks.append(
            specs.storage_gb is not None and specs.storage_gb >= requirements.min_storage_gb
        )
    if requirements.storage_type != "any":
        checks.append(specs.storage_type == requirements.storage_type)
    if requirements.min_screen_inches is not None:
        checks.append(
            specs.screen_inches is not None
            and specs.screen_inches >= requirements.min_screen_inches
        )
    if requirements.max_screen_inches is not None:
        checks.append(
            specs.screen_inches is not None
            and specs.screen_inches <= requirements.max_screen_inches
        )
    if requirements.max_weight_kg is not None:
        checks.append(
            specs.weight_kg is not None and specs.weight_kg <= requirements.max_weight_kg
        )
    if requirements.min_battery_hours is not None:
        checks.append(
            specs.battery_hours is not None
            and specs.battery_hours >= requirements.min_battery_hours
        )
    if requirements.required_os != "any":
        checks.append(specs.os == requirements.required_os)
    if requirements.dedicated_gpu_required:
        checks.append(specs.dedicated_gpu)
    if requirements.touchscreen_required:
        checks.append(specs.touchscreen)
    if requirements.preferred_brands:
        checks.append(candidate.product.brand in requirements.preferred_brands)

    if not checks:
        # Nothing specific was asked for; treat as neutral rather than perfect.
        return Decimal("0.5")
    return _clamp(Decimal(sum(1 for passed in checks if passed)) / Decimal(len(checks)))


#: Use cases where a dedicated GPU is genuinely worth paying for. Crediting it
#: everywhere would rank gaming laptops highly for office and development work,
#: where the GPU is cost the user does not benefit from.
_GPU_RELEVANT: frozenset[UseCase] = frozenset(
    {UseCase.GAMING, UseCase.DATA_SCIENCE, UseCase.CONTENT_CREATION}
)


def _value_for_money(
    candidate: ProductCandidate, requirements: LaptopRequirements
) -> Decimal:
    """How much *useful* capability the effective price buys.

    Uses the effective price — the amount actually paid — not the list price, and
    never includes cashback, which is not a checkout reduction.
    """
    effective = candidate.price.effective_price.amount
    if effective <= 0:
        return _ZERO
    specs = candidate.product.specs
    gpu_counts = (
        requirements.use_case in _GPU_RELEVANT or requirements.dedicated_gpu_required
    )
    # Unknown specs contribute nothing, so an under-described listing cannot
    # score as though it were well equipped.
    capability = (
        (_dec(specs.ram_gb) or _ZERO) * Decimal("1.5")
        + (_dec(specs.storage_gb) or _ZERO) / Decimal("32")
        + (Decimal("24") if specs.dedicated_gpu and gpu_counts else _ZERO)
        + (_dec(specs.battery_hours) or _ZERO)
    )
    if capability <= 0:
        return _ZERO
    # Normalised against a reference of one capability point per INR 1,200.
    return _ratio(capability * Decimal("1200") / effective, Decimal("1.6"))


def _budget_headroom(
    candidate: ProductCandidate, requirements: LaptopRequirements
) -> Decimal:
    """Reward staying comfortably inside budget, without rewarding cheapness.

    A candidate at 100% of budget scores 0; one at 60% or below scores 1. Below
    that there is no additional credit, so an underpowered bargain does not win
    on price alone.
    """
    budget = requirements.budget_max
    if budget is None:
        return Decimal("0.5")
    if budget.amount <= 0:
        return _ZERO
    used = candidate.price.effective_price.amount / budget.amount
    if used >= _ONE:
        return _ZERO
    headroom = (_ONE - used) / Decimal("0.4")
    return _clamp(headroom)


def _spec_strength(
    candidate: ProductCandidate, requirements: LaptopRequirements
) -> Decimal:
    """Absolute specification strength, weighted for the stated use case."""
    specs = candidate.product.specs
    weight = _dec(specs.weight_kg)
    attributes = {
        "ram": _credit(specs.ram_gb, _RAM_CEILING),
        "storage": _credit(specs.storage_gb, _STORAGE_CEILING),
        "battery": _credit(specs.battery_hours, _BATTERY_CEILING),
        "portability": _ZERO
        if weight is None
        else _clamp((Decimal("2.5") - weight) / Decimal("1.2")),
        "gpu": _ONE if specs.dedicated_gpu else Decimal("0.2"),
        "refresh": _credit(specs.refresh_rate_hz, Decimal("165")),
    }
    profile = _SPEC_PROFILE[requirements.use_case]
    total = sum(
        (attributes[name] * weight for name, weight in profile), start=_ZERO
    )
    return _clamp(total)


def _rating_confidence(candidate: ProductCandidate) -> Decimal:
    """Rating, discounted when it rests on very few reviews.

    An unrated listing scores near zero rather than mid-range. On a live
    marketplace, no rating usually means a new or grey-market seller, and
    treating that as merely "average" let unrated no-name listings outrank
    well-reviewed ones purely on price.
    """
    rating = candidate.product.rating
    if rating is None:
        return Decimal("0.1")
    normalised = _ratio(Decimal(str(rating)), Decimal("5"))
    # Full confidence at 500+ ratings; scaled below that.
    confidence = _ratio(Decimal(candidate.product.rating_count), Decimal("500"))
    floor = Decimal("0.4")
    return _clamp(normalised * (floor + (_ONE - floor) * confidence))
