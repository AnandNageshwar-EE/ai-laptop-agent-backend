"""Structured output schemas.

Every LLM response is one of these models. There is no code path that reads free
text from the model and interprets it, and no use of ``eval``, ``exec`` or any
dynamic execution anywhere in this application.

Two design rules apply throughout:

1. **No monetary fields.** Not one of these schemas has a price, discount or
   total. Prices come from marketplace data through the pricing code, so a model
   has no channel to introduce a figure into the result.
2. **Closed enums and bounded strings.** An unexpected value is a validation
   error at the boundary rather than data that reaches graph state.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..domain.enums import Bank, UseCase

StorageTypeOut = Literal["ssd", "hdd", "any"]
OperatingSystemOut = Literal["windows", "macos", "linux", "chromeos", "any"]
Confidence = Literal["low", "medium", "high"]


class ExtractedBudget(BaseModel):
    """Budget as a bare number plus a currency code.

    This is the one number a model may produce, because the user stated it. It is
    re-validated into :class:`~laptop_agent.domain.money.Money` and is only ever
    used as a *constraint*, never as a product price.
    """

    model_config = ConfigDict(extra="forbid")

    amount: Annotated[float, Field(gt=0, le=100_000_000)]
    currency: Literal["INR", "USD"] = "INR"


class RequirementExtraction(BaseModel):
    """Requirements parsed from a shopping request."""

    model_config = ConfigDict(extra="forbid")

    use_case: UseCase = UseCase.GENERAL
    budget_max: ExtractedBudget | None = None
    budget_min: ExtractedBudget | None = None

    min_ram_gb: Annotated[int, Field(ge=2, le=256)] | None = None
    min_storage_gb: Annotated[int, Field(ge=32, le=8192)] | None = None
    storage_type: StorageTypeOut = "any"
    min_screen_inches: Annotated[float, Field(ge=8.0, le=20.0)] | None = None
    max_screen_inches: Annotated[float, Field(ge=8.0, le=20.0)] | None = None
    max_weight_kg: Annotated[float, Field(gt=0.2, le=6.0)] | None = None
    min_battery_hours: Annotated[float, Field(ge=1.0, le=30.0)] | None = None
    required_os: OperatingSystemOut = "any"
    dedicated_gpu_required: bool = False
    touchscreen_required: bool = False

    preferred_brands: Annotated[list[str], Field(max_length=8)] = Field(default_factory=list)
    excluded_brands: Annotated[list[str], Field(max_length=8)] = Field(default_factory=list)

    #: Names of fields the user stated as non-negotiable.
    mandatory_fields: Annotated[list[str], Field(max_length=14)] = Field(default_factory=list)

    #: Bank eligibility only. There is deliberately no field for card digits.
    eligible_banks: Annotated[list[Bank], Field(max_length=8)] = Field(default_factory=list)
    has_exchange_device: bool = False
    wants_no_cost_emi: bool = False

    #: Set when the request contained text aimed at the model rather than
    #: shopping requirements. Cross-checks the deterministic scan.
    contains_suspicious_instructions: bool = False
    confidence: Confidence = "medium"

    @field_validator("preferred_brands", "excluded_brands")
    @classmethod
    def _clean(cls, values: list[str]) -> list[str]:
        return [value.strip().lower()[:32] for value in values if value.strip()]


class ClarificationDecision(BaseModel):
    """Whether to ask the user a question before searching."""

    model_config = ConfigDict(extra="forbid")

    needs_clarification: bool
    question: Annotated[str, Field(max_length=240)] = ""
    #: Which missing fields the question covers.
    missing_fields: Annotated[list[str], Field(max_length=4)] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _no_sensitive_ask(cls, value: str) -> str:
        """Refuse a question that solicits payment or identity data."""
        lowered = value.lower()
        forbidden = (
            "card number", "cvv", "expiry", "otp", "password", "pin ",
            "aadhaar", "pan number", "account number", "full name",
            "address", "phone number", "email",
        )
        if any(term in lowered for term in forbidden):
            raise ValueError("clarification question must not request sensitive data")
        return value.strip()


class SearchDecision(BaseModel):
    """Search terms per marketplace, derived from the requirements."""

    model_config = ConfigDict(extra="forbid")

    #: Keyword query only. There is no URL, host or path field — a model cannot
    #: influence where the request goes, only what is searched for.
    query: Annotated[str, Field(min_length=2, max_length=120)]
    #: Alternate phrasings, tried when the primary query is too narrow.
    alternate_queries: Annotated[list[str], Field(max_length=3)] = Field(default_factory=list)
    rationale: Annotated[str, Field(max_length=280)] = ""


class RunnerUpNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9._-]{4,64}$")]
    why_not: Annotated[str, Field(max_length=240)]


class TradeOffOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Annotated[str, Field(min_length=2, max_length=48)]
    detail: Annotated[str, Field(min_length=2, max_length=280)]


class RecommendationExplanation(BaseModel):
    """Prose for an already-selected recommendation.

    Note what is absent: no product id for the winner, no price, no score. The
    model cannot change which product is recommended, and cannot state a figure.
    """

    model_config = ConfigDict(extra="forbid")

    rationale: Annotated[str, Field(min_length=10, max_length=1200)]
    trade_offs: Annotated[list[TradeOffOut], Field(max_length=6)] = Field(default_factory=list)

    @field_validator("trade_offs", mode="before")
    @classmethod
    def _accept_plain_strings(cls, value: object) -> object:
        """Normalise a bare string trade-off into the structured form.

        Observed in production against anthropic/claude-opus-5 via OpenRouter:
        roughly one call in eight returned ``trade_offs`` as a list of sentences
        rather than objects (``trade_offs.0:model_type``), which failed both
        attempts and dropped the request to the deterministic explainer.

        This is normalisation, not blind parsing: the string still goes through
        the same length bounds as any other detail, and the monetary-claim screen
        runs over it afterwards regardless. Rejecting a usable answer over a
        container shape was the worse trade.
        """
        if not isinstance(value, list):
            return value
        normalised: list[object] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue
                normalised.append({"dimension": "trade-off", "detail": text[:280]})
            else:
                normalised.append(item)
        return normalised
    #: Must not be smaller than the number of runner-ups the node actually
    #: sends. It was capped at 4 while the node grew to 5, so a correct model
    #: response was rejected as "too_long" and the whole call was wasted.
    runner_up_notes: Annotated[list[RunnerUpNote], Field(max_length=5)] = Field(
        default_factory=list
    )


class ScopeAssessment(BaseModel):
    """Advisory second opinion for genuinely ambiguous scope cases.

    Advisory only: it can confirm an in-scope request but can never overturn a
    deterministic block.
    """

    model_config = ConfigDict(extra="forbid")

    is_laptop_shopping_related: bool
    reason: Annotated[str, Field(max_length=200)] = ""
