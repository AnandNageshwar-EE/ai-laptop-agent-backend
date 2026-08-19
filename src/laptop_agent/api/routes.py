"""HTTP routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ..agent import AgentReply, LaptopAgent, get_agent
from ..config import get_settings
from ..domain.recommendation import Recommendation
from ..prompts.provider import get_prompt_provider
from ..domain.product import LaptopSpecs
from ..guardrails.display import neutralise_for_display
from .schemas import (
    AppliedOfferView,
    CandidateRowView,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RecommendationView,
    SpecsView,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness plus the configuration facts worth confirming at a glance.

    Reports whether prompt caching is on and the stable-prefix fingerprint, so a
    deployment that accidentally changed the cached prefix is visible here rather
    than only as a silently collapsed cache hit rate.
    """
    settings = get_settings()
    agent = get_agent()
    return HealthResponse(
        status="ok",
        application=settings.application,
        environment=settings.environment,
        graph_version=settings.graph_version,
        llm_mode=settings.llm_mode,
        model=settings.traced_model_name,
        tracing_enabled=settings.tracing_enabled,
        prompt_cache_enabled=settings.prompt_caching_enabled,
        provider_prompt_caching=settings.provider_prompt_caching_supported,
        prompt_prefix_fingerprint=get_prompt_provider().prefix_fingerprint(),
        marketplaces=[market.value for market in agent.registry.marketplaces],
        marketplace_source=settings.effective_marketplace_source,
        live_prices=settings.live_marketplace_enabled,
    )


@router.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(request: ChatRequest, agent: LaptopAgent = Depends(get_agent)) -> ChatResponse:
    """One conversational turn."""
    reply = agent.run(message=request.message, session_id=request.session_id)
    return _to_response(reply)


def _to_response(reply: AgentReply) -> ChatResponse:
    view = _to_view(reply.recommendation)
    if view is not None:
        flagged = reply.diagnostics.get("injection_flags", 0)
        view = view.model_copy(update={"flagged_listing_count": int(flagged or 0)})
    return ChatResponse(
        session_id=reply.session_id,
        response_text=reply.response_text,
        blocked=reply.blocked,
        block_reason=reply.block_reason,
        awaiting_clarification=reply.awaiting_clarification,
        clarification_question=reply.clarification_question,
        recommendation=view,
        trade_off_required=reply.trade_off_required,
        warnings=reply.warnings,
        diagnostics=reply.diagnostics,
    )


_OS_LABELS = {
    "windows": "Windows",
    "macos": "macOS",
    "linux": "Linux",
    "chromeos": "ChromeOS",
}

_OFFER_LABELS = {
    "upfront_discount": "Upfront discount",
    "bank_discount": "Bank card discount",
    "exchange_bonus": "Exchange bonus",
    "coupon": "Coupon",
    "cashback": "Cashback",
    "no_cost_emi": "No-cost EMI",
}


#: Shown when a marketplace listing does not report a specification. Displaying
#: "Not stated" is correct; inventing a plausible value would not be.
UNKNOWN = "Not stated"


def _to_specs_view(specs: LaptopSpecs) -> SpecsView:
    """Render specs for display. Formatting only — no values are derived."""
    if specs.storage_gb is None:
        storage = UNKNOWN
    else:
        unit = (
            f"{specs.storage_gb // 1024}TB"
            if specs.storage_gb >= 1024
            else f"{specs.storage_gb}GB"
        )
        storage = f"{unit} {specs.storage_type.upper()}" if specs.storage_type else unit

    if specs.screen_inches is None:
        display = UNKNOWN
    else:
        refresh = (
            f" · {specs.refresh_rate_hz}Hz"
            if specs.refresh_rate_hz and specs.refresh_rate_hz > 60
            else ""
        )
        display = f'{specs.screen_inches}"{refresh}'

    return SpecsView(
        ram=f"{specs.ram_gb}GB" if specs.ram_gb is not None else UNKNOWN,
        storage=storage,
        cpu=specs.cpu or UNKNOWN,
        graphics=specs.gpu or ("Dedicated" if specs.dedicated_gpu else UNKNOWN),
        dedicated_gpu=specs.dedicated_gpu,
        display=display,
        weight=f"{specs.weight_kg} kg" if specs.weight_kg is not None else UNKNOWN,
        battery=f"{specs.battery_hours} hrs" if specs.battery_hours is not None else UNKNOWN,
        os=_OS_LABELS.get(specs.os, specs.os) if specs.os else UNKNOWN,
        touchscreen=specs.touchscreen,
    )


def _to_row(runner: Any) -> CandidateRowView:
    """Shape one candidate for a comparison card."""
    return CandidateRowView(
        product_id=runner.product_id,
        marketplace=runner.marketplace.value,
        title=neutralise_for_display(runner.title)[0],
        brand=runner.brand,
        url=str(runner.url),
        rating=runner.rating,
        rating_count=runner.rating_count,
        listed_price=str(runner.listed_price),
        effective_price=str(runner.effective_price),
        upfront_savings=str(runner.upfront_savings),
        cashback_value=str(runner.cashback_value),
        has_discount=runner.upfront_savings.amount > 0,
        has_cashback=runner.cashback_value.amount > 0,
        unmet_conditional_offer_count=len(runner.unmet_conditional_offers),
        score=float(runner.score),
        specs=_to_specs_view(runner.specs),
        why_not=runner.why_not,
    )


def _to_view(recommendation: Recommendation | None) -> RecommendationView | None:
    if recommendation is None:
        return None
    return RecommendationView(
        product_id=recommendation.product_id,
        marketplace=recommendation.marketplace.value,
        # Neutralised for display only. The validator already compared the exact
        # provider bytes, so cleaning here cannot weaken provenance.
        title=neutralise_for_display(recommendation.title)[0],
        brand=recommendation.brand,
        url=str(recommendation.url),
        rating=recommendation.rating,
        rating_count=recommendation.rating_count,
        specs=_to_specs_view(recommendation.specs),
        listed_price=str(recommendation.listed_price),
        effective_price=str(recommendation.effective_price),
        upfront_savings=str(recommendation.upfront_savings),
        cashback_value=str(recommendation.cashback_value),
        has_cashback=recommendation.cashback_value.amount > 0,
        has_discount=recommendation.upfront_savings.amount > 0,
        applied_offers=[
            AppliedOfferView(
                offer_id=offer.offer_id,
                kind=_OFFER_LABELS.get(offer.kind.value, offer.kind.value),
                amount=str(offer.amount),
                reason=offer.reason,
            )
            for offer in recommendation.applied_offers
        ],
        unmet_conditional_offer_count=len(recommendation.unmet_conditional_offers),
        score=float(recommendation.score),
        scoring_version=recommendation.scoring_version,
        rationale=recommendation.rationale,
        trade_offs=[
            {"dimension": item.dimension, "detail": item.detail}
            for item in recommendation.trade_offs
        ],
        near_budget_alternatives=[
            _to_row(runner) for runner in recommendation.near_budget_alternatives
        ],
        runner_ups=[_to_row(runner) for runner in recommendation.runner_ups],
        warnings=recommendation.warnings,
    )
