"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..agent import AgentReply, LaptopAgent, get_agent
from ..config import get_settings
from ..domain.recommendation import Recommendation
from ..prompts.provider import PromptTask, get_prompt_provider
from .schemas import ChatRequest, ChatResponse, HealthResponse, RecommendationView

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
        prompt_prefix_fingerprint=get_prompt_provider().prefix_fingerprint(),
        marketplaces=[market.value for market in agent.registry.marketplaces],
    )


@router.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(request: ChatRequest, agent: LaptopAgent = Depends(get_agent)) -> ChatResponse:
    """One conversational turn."""
    reply = agent.run(message=request.message, session_id=request.session_id)
    return _to_response(reply)


def _to_response(reply: AgentReply) -> ChatResponse:
    return ChatResponse(
        session_id=reply.session_id,
        response_text=reply.response_text,
        blocked=reply.blocked,
        block_reason=reply.block_reason,
        awaiting_clarification=reply.awaiting_clarification,
        clarification_question=reply.clarification_question,
        recommendation=_to_view(reply.recommendation),
        trade_off_required=reply.trade_off_required,
        warnings=reply.warnings,
        diagnostics=reply.diagnostics,
    )


def _to_view(recommendation: Recommendation | None) -> RecommendationView | None:
    if recommendation is None:
        return None
    return RecommendationView(
        product_id=recommendation.product_id,
        marketplace=recommendation.marketplace.value,
        title=recommendation.title,
        url=str(recommendation.url),
        listed_price=str(recommendation.listed_price),
        effective_price=str(recommendation.effective_price),
        upfront_savings=str(recommendation.upfront_savings),
        cashback_value=str(recommendation.cashback_value),
        unmet_conditional_offer_count=len(recommendation.unmet_conditional_offers),
        score=float(recommendation.score),
        scoring_version=recommendation.scoring_version,
        rationale=recommendation.rationale,
        trade_offs=[
            {"dimension": item.dimension, "detail": item.detail}
            for item in recommendation.trade_offs
        ],
        runner_ups=[
            {
                "product_id": runner.product_id,
                "marketplace": runner.marketplace.value,
                "title": runner.title,
                "url": str(runner.url),
                "effective_price": str(runner.effective_price),
                "why_not": runner.why_not,
            }
            for runner in recommendation.runner_ups
        ],
        warnings=recommendation.warnings,
    )
