"""Single entry point for reasoning, hiding the live/offline distinction.

Nodes call this and never branch on mode. Both paths return the same validated
Pydantic models, so the graph, the guardrails and the tests are identical either
way.

The live path is also where the prompt-cache layout is applied: the system
message carries the stable, cache-marked blocks from the prompt provider, and all
per-request data goes into the human message *after* those breakpoints. Untrusted
content is wrapped, never concatenated into the system prompt.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings
from ..domain.product import ProductCandidate
from ..domain.requirements import LaptopRequirements
from ..guardrails.untrusted import TrustLabel, wrap_untrusted
from ..prompts.provider import CachedPromptProvider, PromptTask, get_prompt_provider
from .offline import OfflineReasoner
from .schemas import (
    ClarificationDecision,
    RecommendationExplanation,
    RequirementExtraction,
    SearchDecision,
)
from .structured import InvocationStats, StructuredLLM


class Reasoner:
    """Mode-agnostic reasoning facade."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        prompts: CachedPromptProvider | None = None,
        chat_model: Any | None = None,
    ) -> None:
        self._settings = get_settings() if settings is None else settings
        self._prompts = prompts or get_prompt_provider()
        self._offline = OfflineReasoner()
        self._structured: StructuredLLM | None = None

        if self._settings.llm_mode == "live":
            model = chat_model
            if model is None:
                from .provider import build_chat_model

                model = build_chat_model(self._settings)
            self._structured = StructuredLLM(
                model, max_retries=self._settings.llm_structured_retries
            )

    @property
    def is_live(self) -> bool:
        return self._structured is not None

    def prompt_metadata(self, task: PromptTask) -> dict[str, str]:
        return self._prompts.versions(task)

    # ------------------------------------------------------------------

    def extract_requirements(
        self,
        text: str,
        *,
        previous: RequirementExtraction | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[RequirementExtraction, InvocationStats | None]:
        if self._structured is None:
            return self._offline.extract_requirements(text, previous=previous), None

        payload: dict[str, Any] = {"shopping_request": text}
        if previous is not None:
            payload["requirements_so_far"] = previous.model_dump(mode="json")
        messages = self._messages(
            PromptTask.REQUIREMENTS,
            wrap_untrusted(payload, TrustLabel.USER_INPUT),
        )
        return self._structured.invoke(RequirementExtraction, messages, config=config)

    def decide_clarification(
        self,
        requirements: LaptopRequirements,
        missing: list[str],
        *,
        config: dict[str, Any] | None = None,
    ) -> tuple[ClarificationDecision, InvocationStats | None]:
        if self._structured is None:
            return self._offline.decide_clarification(requirements, missing), None

        messages = self._messages(
            PromptTask.CLARIFICATION,
            wrap_untrusted(
                {
                    "requirements": requirements.model_dump(mode="json"),
                    "missing_fields": missing,
                },
                TrustLabel.CONVERSATION_STATE,
            ),
        )
        return self._structured.invoke(ClarificationDecision, messages, config=config)

    def decide_search(self, requirements: LaptopRequirements) -> SearchDecision:
        """Search terms are derived deterministically in both modes.

        There is no upside to a model call here — the requirements are already
        structured — and keeping it deterministic means the provider cache key is
        stable across identical requests.
        """
        return self._offline.decide_search(requirements)

    def explain_recommendation(
        self,
        winner: ProductCandidate,
        runner_ups: list[ProductCandidate],
        requirements: LaptopRequirements,
        *,
        config: dict[str, Any] | None = None,
    ) -> tuple[RecommendationExplanation, InvocationStats | None]:
        if self._structured is None:
            return (
                self._offline.explain_recommendation(winner, runner_ups, requirements),
                None,
            )

        messages = self._messages(
            PromptTask.RECOMMENDATION,
            wrap_untrusted(
                {
                    "requirements": requirements.model_dump(mode="json"),
                    "selected_candidate": _candidate_view(winner),
                    "runner_up_candidates": [_candidate_view(c) for c in runner_ups],
                },
                TrustLabel.MARKETPLACE_DATA,
            ),
        )
        return self._structured.invoke(
            RecommendationExplanation, messages, config=config
        )

    # ------------------------------------------------------------------

    def _messages(self, task: PromptTask, human_content: str) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        # System = stable, cache-marked blocks only. Human = everything dynamic.
        return [
            SystemMessage(content=self._prompts.system_blocks(task)),
            HumanMessage(content=human_content),
        ]


def _candidate_view(candidate: ProductCandidate) -> dict[str, Any]:
    """What the model is shown about a candidate.

    Prices are included as *rendered strings* rather than numbers, and are
    labelled, so the model can reason about relative cost without being handed
    figures it might recombine arithmetically. Its output is screened for
    monetary claims regardless.
    """
    specs = candidate.product.specs
    return {
        "product_id": candidate.product.product_id,
        "marketplace": candidate.product.marketplace.value,
        "title": candidate.product.title,
        "brand": candidate.product.brand,
        "rating": candidate.product.rating,
        "rating_count": candidate.product.rating_count,
        "specs": {
            "ram_gb": specs.ram_gb,
            "storage_gb": specs.storage_gb,
            "storage_type": specs.storage_type,
            "cpu": specs.cpu,
            "gpu": specs.gpu,
            "dedicated_gpu": specs.dedicated_gpu,
            "screen_inches": specs.screen_inches,
            "weight_kg": specs.weight_kg,
            "battery_hours": specs.battery_hours,
            "os": specs.os,
            "touchscreen": specs.touchscreen,
            "refresh_rate_hz": specs.refresh_rate_hz,
        },
        "price_description": {
            "effective_price_display": str(candidate.price.effective_price),
            "has_upfront_discount": candidate.price.total_upfront_discount.amount > 0,
            "has_cashback_returned_later": candidate.price.cashback_value.amount > 0,
            "conditional_offers_not_applied": len(candidate.price.unmet_conditional_offers),
        },
        "seller_description_untrusted": candidate.product.description,
    }
