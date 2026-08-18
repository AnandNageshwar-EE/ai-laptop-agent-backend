"""HTTP request/response contracts.

The API is a second trust boundary, independent of the agent's own guardrails:
a request body is validated here before the agent sees it, so a malformed or
oversized payload is rejected by FastAPI with a 422 rather than reaching the
graph. The agent's input guardrail then runs anyway — neither layer relies on
the other.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

#: Mirrors the guardrail ceiling. Kept as a literal so the HTTP contract does not
#: shift when configuration changes; the guardrail remains authoritative.
MAX_MESSAGE_CHARS = 2_000


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]
    #: Omit on the first turn; the server issues one.
    session_id: Annotated[str, Field(pattern=r"^sess_[0-9a-f]{32}$")] | None = None


class SpecsView(BaseModel):
    """Specifications, as validated provider data. Display-ready values."""

    model_config = ConfigDict(extra="forbid")

    ram: str
    storage: str
    cpu: str
    graphics: str
    dedicated_gpu: bool
    display: str
    weight: str
    battery: str
    os: str
    touchscreen: bool


class AppliedOfferView(BaseModel):
    """One discount actually subtracted, so the price shows as a breakdown."""

    model_config = ConfigDict(extra="forbid")

    offer_id: str
    kind: str
    amount: str
    reason: str


class CandidateRowView(BaseModel):
    """A runner-up, shaped for a side-by-side comparison table."""

    model_config = ConfigDict(extra="forbid")

    product_id: str
    marketplace: str
    title: str
    brand: str
    url: str
    rating: float | None = None
    rating_count: int = 0
    listed_price: str
    effective_price: str
    upfront_savings: str
    cashback_value: str
    has_discount: bool = False
    has_cashback: bool = False
    unmet_conditional_offer_count: int = 0
    score: float
    specs: SpecsView
    why_not: str = ""


class RecommendationView(BaseModel):
    """Flattened recommendation for the UI.

    Every monetary value is a pre-rendered string. The frontend displays them
    verbatim — parsing or recomputing one would make the UI a second,
    unvalidated pricing implementation.
    """

    model_config = ConfigDict(extra="forbid")

    product_id: str
    marketplace: str
    title: str
    brand: str
    url: str
    rating: float | None = None
    rating_count: int = 0
    specs: SpecsView

    listed_price: str
    effective_price: str
    upfront_savings: str
    cashback_value: str
    has_cashback: bool
    has_discount: bool
    applied_offers: list[AppliedOfferView] = Field(default_factory=list)
    unmet_conditional_offer_count: int

    score: float
    scoring_version: str
    rationale: str
    trade_offs: list[dict[str, str]] = Field(default_factory=list)
    runner_ups: list[CandidateRowView] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: How many listings were disqualified for attempting to manipulate the agent.
    flagged_listing_count: int = 0


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    response_text: str
    blocked: bool = False
    block_reason: str | None = None
    awaiting_clarification: bool = False
    clarification_question: str = ""
    recommendation: RecommendationView | None = None
    trade_off_required: bool = False
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    application: str
    environment: str
    graph_version: str
    llm_mode: str
    model: str
    tracing_enabled: bool
    prompt_cache_enabled: bool
    prompt_prefix_fingerprint: str
    marketplaces: list[str]
    #: "serpapi" for live listings, "fixtures" for the bundled simulated data.
    marketplace_source: str
    live_prices: bool
