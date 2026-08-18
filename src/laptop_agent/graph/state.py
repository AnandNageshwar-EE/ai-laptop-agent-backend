"""Typed graph state.

Every field is either a primitive or a validated domain model. There is no
``dict[str, Any]`` bucket, and no field holds raw marketplace JSON — the search
nodes write only ``Product`` instances that already passed
:class:`~laptop_agent.guardrails.tool_output.MarketplaceResponseValidator`.

``products``, ``quarantined_products``, ``injection_flags`` and
``marketplaces_used`` carry ``operator.add`` reducers because the marketplace
searches run concurrently and each writes its own slice. Everything else is
written by exactly one node, so last-write-wins is correct.

Provenance note: ``products`` is append-only and is written *only* by the search
nodes. That is what makes it usable as the provenance set the recommendation
validator checks against — see
:class:`~laptop_agent.guardrails.recommendation_validator.ProviderRegistry`.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..domain.product import Offer, Product, ProductCandidate
from ..domain.recommendation import Recommendation
from ..domain.requirements import LaptopRequirements, PurchaseProfile
from ..domain.scoring import ProductScore


class LaptopAgentState(TypedDict, total=False):
    """State threaded through the LangGraph workflow."""

    # ----- identity -----
    session_id: str
    turn_count: int

    # ----- input (already sanitised by the input guardrail) -----
    user_request: str

    # ----- guardrail outcome -----
    blocked: bool
    block_reason: str | None
    #: Canned, detail-free reply used when blocked.
    safe_response: str

    # ----- requirements -----
    requirements: LaptopRequirements | None
    purchase_profile: PurchaseProfile | None
    requirement_confidence: str

    # ----- clarification -----
    needs_clarification: bool
    clarification_question: str
    clarification_fields: list[str]

    # ----- search -----
    search_query: str
    products: Annotated[list[Product], operator.add]
    quarantined_products: Annotated[list[tuple[str, str]], operator.add]
    #: (product_id, categories) for seller text that attempted an injection.
    injection_flags: Annotated[list[tuple[str, list[str]]], operator.add]
    marketplaces_used: Annotated[list[str], operator.add]

    # ----- offers -----
    offers: list[Offer]
    quarantined_offers: list[tuple[str, str]]

    # ----- pricing / ranking -----
    candidates: list[ProductCandidate]
    ranked_products: list[ProductScore]

    # ----- recommendation -----
    recommendation: Recommendation | None
    trade_off_required: bool

    # ----- validation / repair loop -----
    validation_failures: list[str]
    #: "marketplace:product_id" pairs the validator rejected; excluded on re-rank.
    excluded_candidates: list[str]
    repair_attempts: int

    # ----- output -----
    response_text: str
    warnings: list[str]


def initial_state(
    *,
    session_id: str,
    user_request: str,
    requirements: LaptopRequirements | None = None,
    purchase_profile: PurchaseProfile | None = None,
    turn_count: int = 0,
) -> LaptopAgentState:
    """Build a fresh state.

    Note what is *not* carried in: products, offers, candidates or prices. Those
    are always re-derived from live provider data, so a previous turn cannot
    smuggle a stale price or a fabricated candidate into this one.
    """
    return LaptopAgentState(
        session_id=session_id,
        turn_count=turn_count,
        user_request=user_request,
        blocked=False,
        block_reason=None,
        safe_response="",
        requirements=requirements,
        purchase_profile=purchase_profile or PurchaseProfile(),
        requirement_confidence="low",
        needs_clarification=False,
        clarification_question="",
        clarification_fields=[],
        search_query="",
        products=[],
        quarantined_products=[],
        injection_flags=[],
        marketplaces_used=[],
        offers=[],
        quarantined_offers=[],
        candidates=[],
        ranked_products=[],
        recommendation=None,
        trade_off_required=False,
        validation_failures=[],
        excluded_candidates=[],
        repair_attempts=0,
        response_text="",
        warnings=[],
    )


def candidate_key(marketplace: str, product_id: str) -> str:
    return f"{marketplace}:{product_id}"
