"""Graph nodes.

Each node is a small, testable function on ``LaptopAgentState``. Shared rules:

* A node returns only the keys it changed. LangGraph merges them, so a node
  cannot accidentally clear state it did not touch.
* Nothing enters state except validated domain models.
* Every node records its latency, and security-relevant decisions go to the
  audit sink.
* A node never raises for an expected condition (blocked input, no candidates,
  provider outage). It records the condition and lets routing decide.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..audit import AuditEvent, AuditRecord, AuditSink, StructuredLogAuditSink
from ..cache.base import CacheProvider
from ..config import Settings, get_settings
from ..domain.enums import Marketplace, RejectionReason, UseCase
from ..domain.money import Money
from ..domain.product import Offer, Product, ProductCandidate
from ..domain.recommendation import Recommendation, RunnerUp, TradeOff
from ..domain.requirements import LaptopRequirements, PurchaseProfile
from ..guardrails.input_guardrail import InputGuardrail
from ..guardrails.price_claims import screen_price_claims
from ..guardrails.price_validator import PriceValidator
from ..guardrails.recommendation_validator import (
    ProviderRegistry,
    RecommendationValidator,
)
from ..guardrails.scope_guardrail import ConversationStage, ScopeGuardrail
from ..guardrails.tool_input import FetchOffersRequest, SearchProductsRequest
from ..guardrails.tool_output import MarketplaceResponseValidator
from ..llm.facade import Reasoner
from ..llm.structured import StructuredOutputError
from ..marketplace.registry import MarketplaceRegistry, build_registry
from ..observability.metrics import RunMetrics
from ..pricing.calculator import PriceCalculator
from ..ranking.scorer import near_budget_candidates, rank_candidates, score_candidate
from ..security.logging import get_logger
from .state import LaptopAgentState, candidate_key

_logger = get_logger("laptop_agent.graph")

#: Fallback text when nothing can be recommended. Never mentions internals.
NO_RESULTS_RESPONSE = (
    "I could not find a laptop that meets all of your stated requirements from "
    "the marketplaces I can search. Try relaxing one constraint — for example a "
    "slightly higher budget, or less RAM — and I will look again."
)

VALIDATION_FAILED_RESPONSE = (
    "I found candidates but could not verify the pricing and requirement checks "
    "for any of them with enough confidence to recommend one. Rather than show "
    "you a result I cannot stand behind, I would rather try again — please "
    "restate or adjust your requirements."
)


def _timed(node: str, metrics: RunMetrics, func: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    error: str | None = None
    try:
        return func()
    except Exception as exc:  # recorded, then re-raised for the graph to surface
        error = type(exc).__name__
        raise
    finally:
        metrics.record_node(node, round((time.perf_counter() - started) * 1000, 2), error)


class AgentNodes:
    """Holds the collaborators the nodes need. One instance per run."""

    def __init__(
        self,
        *,
        metrics: RunMetrics,
        settings: Settings | None = None,
        registry: MarketplaceRegistry | None = None,
        cache: CacheProvider | None = None,
        reasoner: Reasoner | None = None,
        audit: AuditSink | None = None,
        stage: ConversationStage = ConversationStage.OPENING,
    ) -> None:
        self.settings = get_settings() if settings is None else settings
        self.metrics = metrics
        self.audit = StructuredLogAuditSink() if audit is None else audit
        self.stage = stage
        # `registry or build_registry(...)` would be wrong: MarketplaceRegistry
        # defines __len__, so an empty one is falsy.
        self.registry = (
            build_registry(cache=cache, settings=self.settings)
            if registry is None
            else registry
        )
        self.reasoner = Reasoner(settings=self.settings) if reasoner is None else reasoner
        self.input_guardrail = InputGuardrail(settings=self.settings)
        self.scope_guardrail = ScopeGuardrail()
        self.prices = PriceValidator()
        self.calculator = PriceCalculator(self.prices)
        self.recommendation_validator = RecommendationValidator(self.prices)
        #: Injection categories seen in seller text this run.
        self._content_flags: list[tuple[str, list[str]]] = []

    # ------------------------------------------------------------------
    # 1. input guardrail
    # ------------------------------------------------------------------

    def input_guardrail_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            session_id = state["session_id"]
            raw = state.get("user_request", "")

            result = self.input_guardrail.check(raw)
            if result.blocked:
                self.metrics.guardrail_blocks += 1
                assert result.reason is not None
                event = (
                    AuditEvent.INJECTION_DETECTED
                    if result.reason
                    in {
                        RejectionReason.PROMPT_INJECTION,
                        RejectionReason.SYSTEM_MANIPULATION,
                        RejectionReason.SECRET_EXFILTRATION,
                    }
                    else AuditEvent.INPUT_BLOCKED
                )
                self.audit.record(
                    AuditRecord(
                        event=event,
                        session_id=session_id,
                        node="input_guardrail",
                        reason=result.reason.value,
                        detail=result.internal_detail,
                    )
                )
                return {
                    "blocked": True,
                    "block_reason": result.reason.value,
                    "safe_response": result.user_message,
                    "response_text": result.user_message,
                }

            sanitised = result.value or ""
            scope = self.scope_guardrail.check(sanitised, stage=self.stage)
            if scope.blocked:
                self.metrics.guardrail_blocks += 1
                assert scope.reason is not None
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.SCOPE_REJECTED,
                        session_id=session_id,
                        node="input_guardrail",
                        reason=scope.reason.value,
                        detail=scope.internal_detail,
                    )
                )
                return {
                    "blocked": True,
                    "block_reason": scope.reason.value,
                    "safe_response": scope.user_message,
                    "response_text": scope.user_message,
                }

            warnings = list(state.get("warnings", []))
            if "sensitive_values_removed" in result.notes:
                # Recorded so the audit trail shows PII was stripped, without
                # recording what was stripped.
                warnings.append("Sensitive details were removed and not stored.")

            return {"user_request": sanitised, "blocked": False, "warnings": warnings}

        return _timed("input_guardrail", self.metrics, run)

    # ------------------------------------------------------------------
    # 2. requirements analysis
    # ------------------------------------------------------------------

    def requirements_analysis_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            session_id = state["session_id"]
            text = state.get("user_request", "")

            try:
                # The session's existing requirements go in as context, so a
                # terse follow-up ("2 Lakhs budget") is read as an answer rather
                # than as a whole new request.
                extraction, stats = self.reasoner.extract_requirements(
                    text, known=state.get("requirements")
                )
            except StructuredOutputError as exc:
                # Retries are exhausted. Fail gracefully — never parse free text.
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.LLM_OUTPUT_INVALID,
                        session_id=session_id,
                        node="requirements_analysis",
                        reason=exc.detail[:200],
                        detail={"schema": exc.schema, "attempts": str(exc.attempts)},
                    )
                )
                return {
                    "blocked": True,
                    "block_reason": "llm_output_invalid",
                    "response_text": (
                        "I had trouble interpreting that request. Could you restate "
                        "your laptop requirements — budget, main use, and any must-have "
                        "specifications?"
                    ),
                }

            if stats is not None:
                self.metrics.record_llm(stats)

            if extraction.contains_suspicious_instructions:
                # The deterministic scan already passed this text, so this is the
                # model's independent read. Recorded, not acted on.
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.INJECTION_DETECTED,
                        session_id=session_id,
                        node="requirements_analysis",
                        reason="model_flagged_embedded_instructions",
                    )
                )

            requirements = _to_requirements(extraction, state.get("requirements"))
            profile = _to_profile(extraction, state.get("purchase_profile"))

            return {
                "requirements": requirements,
                "purchase_profile": profile,
                "requirement_confidence": extraction.confidence,
            }

        return _timed("requirements_analysis", self.metrics, run)

    # ------------------------------------------------------------------
    # 3. clarification decision
    # ------------------------------------------------------------------

    def clarification_decision_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            requirements = state.get("requirements")
            if requirements is None:
                return {"needs_clarification": False}

            missing = requirements.missing_for_search()
            if not missing:
                return {"needs_clarification": False}

            decision, stats = self.reasoner.decide_clarification(requirements, missing)
            if stats is not None:
                self.metrics.record_llm(stats)

            if not decision.needs_clarification:
                return {"needs_clarification": False}

            return {
                "needs_clarification": True,
                "clarification_question": decision.question,
                "clarification_fields": decision.missing_fields,
                "response_text": decision.question,
            }

        return _timed("clarification_decision", self.metrics, run)

    # ------------------------------------------------------------------
    # 4. search planning
    # ------------------------------------------------------------------

    def search_planning_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            requirements = state.get("requirements")
            if requirements is None:
                return {"search_query": "laptop"}
            return {"search_query": self.reasoner.decide_search(requirements).query}

        return _timed("search_planning", self.metrics, run)

    # ------------------------------------------------------------------
    # 5. marketplace searches (run concurrently)
    # ------------------------------------------------------------------

    def amazon_search_node(self, state: LaptopAgentState) -> dict[str, Any]:
        return _timed(
            "amazon_search", self.metrics, lambda: self._search(state, Marketplace.AMAZON)
        )

    def flipkart_search_node(self, state: LaptopAgentState) -> dict[str, Any]:
        return _timed(
            "flipkart_search",
            self.metrics,
            lambda: self._search(state, Marketplace.FLIPKART),
        )

    def _search(self, state: LaptopAgentState, marketplace: Marketplace) -> dict[str, Any]:
        session_id = state["session_id"]
        requirements = state.get("requirements")
        node = f"{marketplace.value}_search"

        try:
            request = SearchProductsRequest(
                query=state.get("search_query") or "laptop",
                marketplace=marketplace,
                max_results=self.settings.max_search_results,
                budget_max=requirements.budget_max if requirements else None,
                currency=requirements.currency if requirements else Money(
                    amount=0, currency="INR"
                ).currency,
            )
        except Exception as exc:
            # Invalid tool arguments never reach a provider.
            self.audit.record(
                AuditRecord(
                    event=AuditEvent.TOOL_ARGS_REJECTED,
                    session_id=session_id,
                    node=node,
                    reason=type(exc).__name__,
                    detail={"tool": "search_products"},
                )
            )
            return {"marketplaces_used": []}

        try:
            raw = self.registry.get(marketplace).search(request)
        except Exception as exc:
            # One provider failing must not fail the whole search.
            _logger.warning(
                "marketplace.search_failed",
                extra={"marketplace": marketplace.value, "error_type": type(exc).__name__},
            )
            return {"marketplaces_used": []}

        if raw.get("cache_hit"):
            self.metrics.provider_cache_hits += 1
        else:
            self.metrics.provider_cache_misses += 1

        validator = MarketplaceResponseValidator(marketplace)
        outcome = validator.validate_products(raw)

        for product_id, reason in outcome.quarantined:
            self.audit.record(
                AuditRecord(
                    event=AuditEvent.PRODUCT_QUARANTINED,
                    session_id=session_id,
                    node=node,
                    reason=reason,
                    detail={"product_id": product_id, "marketplace": marketplace.value},
                )
            )
        for product_id, categories in validator.flagged_content:
            self.audit.record(
                AuditRecord(
                    event=AuditEvent.INJECTION_DETECTED,
                    session_id=session_id,
                    node=node,
                    reason="untrusted_marketplace_content",
                    detail={
                        "product_id": product_id,
                        "categories": ",".join(categories),
                        "action": "kept_as_data_and_flagged",
                    },
                )
            )

        self.metrics.products_returned += outcome.accepted_count
        self.metrics.products_quarantined += outcome.quarantined_count
        self.metrics.injection_flags += len(validator.flagged_content)

        return {
            "products": outcome.accepted,
            "quarantined_products": [
                (candidate_key(marketplace.value, pid), reason)
                for pid, reason in outcome.quarantined
            ],
            "injection_flags": validator.flagged_content,
            "marketplaces_used": [marketplace.value],
        }

    # ------------------------------------------------------------------
    # 6. offer analysis
    # ------------------------------------------------------------------

    def offer_analysis_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            session_id = state["session_id"]
            products: list[Product] = state.get("products", [])
            if not products:
                return {"offers": [], "quarantined_offers": []}

            all_offers: list[Offer] = []
            quarantined: list[tuple[str, str]] = []

            by_marketplace: dict[Marketplace, list[Product]] = {}
            for product in products:
                by_marketplace.setdefault(product.marketplace, []).append(product)

            for marketplace in sorted(by_marketplace, key=lambda m: m.value):
                group = by_marketplace[marketplace]
                prices = {product.product_id: product.listed_price for product in group}
                try:
                    request = FetchOffersRequest(
                        marketplace=marketplace, product_ids=sorted(prices)
                    )
                    raw = self.registry.get(marketplace).fetch_offers(request)
                except Exception as exc:
                    _logger.warning(
                        "marketplace.offers_failed",
                        extra={
                            "marketplace": marketplace.value,
                            "error_type": type(exc).__name__,
                        },
                    )
                    continue

                if raw.get("cache_hit"):
                    self.metrics.provider_cache_hits += 1
                else:
                    self.metrics.provider_cache_misses += 1

                validator = MarketplaceResponseValidator(marketplace)
                outcome = validator.validate_offers(raw, known_products=prices)
                all_offers.extend(outcome.accepted)

                for offer_id, reason in outcome.quarantined:
                    quarantined.append((f"{marketplace.value}:{offer_id}", reason))
                    self.audit.record(
                        AuditRecord(
                            event=AuditEvent.OFFER_QUARANTINED,
                            session_id=session_id,
                            node="offer_analysis",
                            reason=reason,
                            detail={"offer_id": offer_id, "marketplace": marketplace.value},
                        )
                    )

            self.metrics.offers_quarantined += len(quarantined)
            return {
                "offers": sorted(all_offers, key=lambda offer: (offer.marketplace.value, offer.offer_id)),
                "quarantined_offers": quarantined,
            }

        return _timed("offer_analysis", self.metrics, run)

    # ------------------------------------------------------------------
    # 7. pricing calculation
    # ------------------------------------------------------------------

    def pricing_calculation_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            session_id = state["session_id"]
            requirements = state.get("requirements")
            if requirements is None:
                return {"candidates": []}

            profile = state.get("purchase_profile") or PurchaseProfile()
            # Listings whose seller text attempted to manipulate the agent are
            # kept as candidates but disqualified from being recommended.
            flagged = {product_id for product_id, _ in state.get("injection_flags", [])}
            candidates, report = self.calculator.build_candidates(
                state.get("products", []),
                state.get("offers", []),
                requirements,
                profile,
                flagged_product_ids=flagged,
            )

            for key, errors in report.price_errors:
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.PRICE_INVALID,
                        session_id=session_id,
                        node="pricing_calculation",
                        reason=errors,
                        detail={"candidate": key},
                    )
                )

            for key in report.trust_flagged:
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.RECOMMENDATION_REJECTED,
                        session_id=session_id,
                        node="pricing_calculation",
                        reason="seller_content_flagged",
                        detail={"candidate": key, "action": "disqualified_from_recommendation"},
                    )
                )

            self.metrics.candidates_built = report.built
            return {
                "candidates": candidates,
                "trust_excluded": report.trust_flagged,
            }

        return _timed("pricing_calculation", self.metrics, run)

    # ------------------------------------------------------------------
    # 8. product ranking
    # ------------------------------------------------------------------

    def product_ranking_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            requirements = state.get("requirements")
            candidates: list[ProductCandidate] = state.get("candidates", [])
            if requirements is None or not candidates:
                return {"ranked_products": []}

            excluded = {
                tuple(key.split(":", 1))
                for key in state.get("excluded_candidates", [])
                if ":" in key
            }
            ranked = rank_candidates(candidates, requirements, exclude=excluded)  # type: ignore[arg-type]

            # A trade-off is required when nothing satisfies every preference.
            trade_off_required = bool(ranked) and all(
                score.total < 0.75 for score in ranked
            )
            return {"ranked_products": ranked, "trade_off_required": trade_off_required}

        return _timed("product_ranking", self.metrics, run)

    # ------------------------------------------------------------------
    # 9. recommendation generation
    # ------------------------------------------------------------------

    def recommendation_generation_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            session_id = state["session_id"]
            requirements = state.get("requirements")
            ranked = state.get("ranked_products", [])
            candidates: list[ProductCandidate] = state.get("candidates", [])
            if requirements is None or not ranked:
                return {"recommendation": None}

            by_key = {candidate.key: candidate for candidate in candidates}
            winner = by_key.get(ranked[0].key)
            if winner is None:
                return {"recommendation": None}
            # Up to five results total, so the UI can list them as ranked cards.
            runner_ups = [
                by_key[score.key] for score in ranked[1:6] if score.key in by_key
            ]

            try:
                explanation, stats = self.reasoner.explain_recommendation(
                    winner, runner_ups, requirements
                )
            except StructuredOutputError as exc:
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.LLM_OUTPUT_INVALID,
                        session_id=session_id,
                        node="recommendation_generation",
                        reason=exc.detail[:200],
                        detail={"schema": exc.schema, "attempts": str(exc.attempts)},
                    )
                )
                # Degrade to a deterministic explanation rather than failing the
                # request: the recommendation itself is computed, not generated.
                explanation = self.reasoner._offline.explain_recommendation(
                    winner, runner_ups, requirements
                )
                stats = None

            if stats is not None:
                self.metrics.record_llm(stats)

            # ---- screen model prose for invented monetary claims ----
            allowed = [
                winner.price.effective_price,
                winner.price.listed_price,
                winner.price.total_upfront_discount,
                winner.price.cashback_value,
            ]
            if requirements.budget_max is not None:
                allowed.append(requirements.budget_max)

            rationale_report = screen_price_claims(explanation.rationale, allowed)
            trade_offs: list[TradeOff] = []
            stripped = list(rationale_report.removed)
            for item in explanation.trade_offs:
                detail_report = screen_price_claims(item.detail, allowed)
                stripped.extend(detail_report.removed)
                trade_offs.append(
                    TradeOff(dimension=item.dimension, detail=detail_report.text)
                )
            if stripped:
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.PRICE_CLAIM_STRIPPED,
                        session_id=session_id,
                        node="recommendation_generation",
                        reason="unauthorised_monetary_claim_in_model_prose",
                        detail={"claims_removed": str(len(stripped))},
                    )
                )

            notes = {note.product_id: note.why_not for note in explanation.runner_up_notes}
            runner_up_models = [
                RunnerUp(
                    product_id=candidate.product.product_id,
                    marketplace=candidate.product.marketplace,
                    title=candidate.product.title,
                    brand=candidate.product.brand,
                    url=candidate.product.url,
                    rating=candidate.product.rating,
                    rating_count=candidate.product.rating_count,
                    listed_price=candidate.price.listed_price,
                    effective_price=candidate.price.effective_price,
                    upfront_savings=candidate.price.total_upfront_discount,
                    cashback_value=candidate.price.cashback_value,
                    unmet_conditional_offers=candidate.price.unmet_conditional_offers,
                    specs=candidate.product.specs,
                    score=score.total,
                    why_not=screen_price_claims(
                        notes.get(candidate.product.product_id, ""), allowed
                    ).text[:280],
                )
                for candidate, score in zip(runner_ups, ranked[1:6])
            ]

            # Options just over the stated budget. Shown, never recommended.
            near_budget = [
                RunnerUp(
                    product_id=candidate.product.product_id,
                    marketplace=candidate.product.marketplace,
                    title=candidate.product.title,
                    brand=candidate.product.brand,
                    url=candidate.product.url,
                    rating=candidate.product.rating,
                    rating_count=candidate.product.rating_count,
                    listed_price=candidate.price.listed_price,
                    effective_price=candidate.price.effective_price,
                    upfront_savings=candidate.price.total_upfront_discount,
                    cashback_value=candidate.price.cashback_value,
                    unmet_conditional_offers=candidate.price.unmet_conditional_offers,
                    specs=candidate.product.specs,
                    score=score.total,
                    why_not="Above your stated budget, listed for comparison only.",
                )
                for candidate, score in near_budget_candidates(candidates, requirements)
            ]

            # Every price field is copied from the validated breakdown. The model
            # contributed prose only.
            recommendation = Recommendation(
                product_id=winner.product.product_id,
                marketplace=winner.product.marketplace,
                title=winner.product.title,
                brand=winner.product.brand,
                url=winner.product.url,
                rating=winner.product.rating,
                rating_count=winner.product.rating_count,
                specs=winner.product.specs,
                listed_price=winner.price.listed_price,
                effective_price=winner.price.effective_price,
                upfront_savings=winner.price.total_upfront_discount,
                cashback_value=winner.price.cashback_value,
                unmet_conditional_offers=winner.price.unmet_conditional_offers,
                applied_offers=winner.price.applied_offers,
                score=ranked[0].total,
                scoring_version=ranked[0].scoring_version,
                rationale=rationale_report.text or "This option matches your stated requirements.",
                trade_offs=trade_offs,
                runner_ups=runner_up_models,
                near_budget_alternatives=near_budget,
                warnings=[
                    *unmet_preferences(winner.product, requirements),
                    *winner.price.warnings,
                ],
            )
            return {"recommendation": recommendation}

        return _timed("recommendation_generation", self.metrics, run)

    # ------------------------------------------------------------------
    # 10. recommendation validation
    # ------------------------------------------------------------------

    def recommendation_validation_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            session_id = state["session_id"]
            recommendation = state.get("recommendation")
            requirements = state.get("requirements")
            profile = state.get("purchase_profile") or PurchaseProfile()

            self.metrics.validation_attempts += 1

            # Provenance set, rebuilt from the append-only products slice that
            # only the search nodes write.
            registry = ProviderRegistry()
            registry.register_all(state.get("products", []))

            if requirements is None:
                return {"validation_failures": ["no_requirements"], "recommendation": None}

            result = self.recommendation_validator.validate(
                recommendation=recommendation,
                candidates=state.get("candidates", []),
                ranked=state.get("ranked_products", []),
                requirements=requirements,
                profile=profile,
                registry=registry,
                rescore=score_candidate,
            )

            if result.is_valid:
                assert recommendation is not None
                self.audit.record(
                    AuditRecord(
                        event=AuditEvent.RECOMMENDATION_APPROVED,
                        session_id=session_id,
                        node="recommendation_validation",
                        detail={
                            "candidate": candidate_key(
                                recommendation.marketplace.value, recommendation.product_id
                            ),
                            "score": str(recommendation.score),
                        },
                    )
                )
                return {
                    "validation_failures": [],
                    "response_text": _render_recommendation(recommendation, state),
                }

            failures = [failure.value for failure in result.failures]
            self.metrics.validation_failures.extend(failures)
            self.audit.record(
                AuditRecord(
                    event=AuditEvent.RECOMMENDATION_REJECTED,
                    session_id=session_id,
                    node="recommendation_validation",
                    reason=result.rejected_reason,
                    detail=result.detail,
                )
            )

            excluded = list(state.get("excluded_candidates", []))
            if recommendation is not None:
                key = candidate_key(
                    recommendation.marketplace.value, recommendation.product_id
                )
                if key not in excluded:
                    excluded.append(key)

            return {
                "validation_failures": failures,
                "excluded_candidates": excluded,
                "repair_attempts": state.get("repair_attempts", 0) + 1,
                "recommendation": None,
            }

        return _timed("recommendation_validation", self.metrics, run)

    # ------------------------------------------------------------------
    # 11. terminal responses
    # ------------------------------------------------------------------

    def no_results_node(self, state: LaptopAgentState) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            constraints = state.get("requirements")
            text = NO_RESULTS_RESPONSE
            if constraints is not None and constraints.mandatory_fields:
                text += (
                    "\n\nThe requirements I treated as non-negotiable were: "
                    + ", ".join(constraints.mandatory_fields).replace("_", " ")
                    + "."
                )
            return {"response_text": text, "recommendation": None}

        # Timed like every other node. These two terminal nodes were originally
        # left unwrapped, so they executed correctly but never appeared in
        # node_latencies_ms — which made the metrics look as though the
        # decline paths were dead code.
        return _timed("no_results", self.metrics, run)

    def validation_failed_node(self, state: LaptopAgentState) -> dict[str, Any]:
        return _timed(
            "validation_failed",
            self.metrics,
            lambda: {"response_text": VALIDATION_FAILED_RESPONSE, "recommendation": None},
        )


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


#: What "the user did not tell us" looks like, per field.
#:
#: This map is the fix for a real defect. The merge previously treated only
#: ``None``, ``[]`` and ``"any"`` as unset — but ``UseCase.GENERAL`` and ``False``
#: are *also* defaults meaning "not specified", so a follow-up turn like
#: "2 Lakhs budget" extracted ``use_case=GENERAL, dedicated_gpu_required=False``
#: and those defaults overwrote the gaming and GPU requirements gathered on the
#: first turn. The agent then searched for a generic laptop and recommended a
#: 2012-era machine for an AI/ML request.
_UNSET_BY_FIELD: dict[str, object] = {
    "use_case": UseCase.GENERAL,
    "storage_type": "any",
    "required_os": "any",
    "dedicated_gpu_required": False,
    "touchscreen_required": False,
    "preferred_brands": [],
    "excluded_brands": [],
    "mandatory_fields": [],
}


def _is_unset(field: str, value: object) -> bool:
    """Whether ``value`` carries no information for ``field``."""
    if value is None:
        return True
    sentinel = _UNSET_BY_FIELD.get(field)
    if sentinel is None:
        return False
    # StrEnum compares equal to its value, so GENERAL == "general" holds.
    return value == sentinel


#: Preferences worth reporting when the winner misses them, with how to phrase it.
_PREFERENCE_LABELS: dict[str, str] = {
    "max_weight_kg": "under {want} kg (this one is {got} kg)",
    "min_battery_hours": "at least {want} hours of battery (this one states {got})",
    "min_screen_inches": "at least {want} inches (this one is {got})",
    "max_screen_inches": "no larger than {want} inches (this one is {got})",
    "min_ram_gb": "at least {want} GB of RAM (this one has {got})",
    "min_storage_gb": "at least {want} GB of storage (this one has {got})",
}


def unmet_preferences(product: Product, requirements: LaptopRequirements) -> list[str]:
    """Preferences the winner does not satisfy, phrased for the user.

    Only *non-mandatory* ones — a mandatory miss disqualifies the candidate
    outright. These are the soft asks that lost a trade-off, and saying so is the
    difference between a recommendation and an unexplained one. Asking for a slim
    1.5 kg machine *and* a gaming GPU is close to contradictory; the agent should
    name which half it could not honour rather than quietly dropping it.
    """
    specs = product.specs
    checks: list[tuple[str, object, object, bool]] = [
        ("max_weight_kg", requirements.max_weight_kg, specs.weight_kg,
         specs.weight_kg is not None
         and requirements.max_weight_kg is not None
         and specs.weight_kg > requirements.max_weight_kg),
        ("min_battery_hours", requirements.min_battery_hours, specs.battery_hours,
         specs.battery_hours is not None
         and requirements.min_battery_hours is not None
         and specs.battery_hours < requirements.min_battery_hours),
        ("min_screen_inches", requirements.min_screen_inches, specs.screen_inches,
         specs.screen_inches is not None
         and requirements.min_screen_inches is not None
         and specs.screen_inches < requirements.min_screen_inches),
        ("max_screen_inches", requirements.max_screen_inches, specs.screen_inches,
         specs.screen_inches is not None
         and requirements.max_screen_inches is not None
         and specs.screen_inches > requirements.max_screen_inches),
        ("min_ram_gb", requirements.min_ram_gb, specs.ram_gb,
         specs.ram_gb is not None
         and requirements.min_ram_gb is not None
         and specs.ram_gb < requirements.min_ram_gb),
        ("min_storage_gb", requirements.min_storage_gb, specs.storage_gb,
         specs.storage_gb is not None
         and requirements.min_storage_gb is not None
         and specs.storage_gb < requirements.min_storage_gb),
    ]

    missed: list[str] = []
    for field, want, got, is_missed in checks:
        if field in requirements.mandatory_fields:
            continue  # a mandatory miss would have disqualified this candidate
        if is_missed:
            missed.append(
                "You asked for " + _PREFERENCE_LABELS[field].format(want=want, got=got) + "."
            )
    return missed


def _to_requirements(
    extraction: Any, previous: LaptopRequirements | None
) -> LaptopRequirements:
    """Convert a validated extraction into the domain model.

    Merging with ``previous`` is what lets a clarification answer add to what was
    already known instead of replacing it.
    """
    data: dict[str, Any] = {
        "use_case": extraction.use_case,
        "min_ram_gb": extraction.min_ram_gb,
        "min_storage_gb": extraction.min_storage_gb,
        "storage_type": extraction.storage_type,
        "min_screen_inches": extraction.min_screen_inches,
        "max_screen_inches": extraction.max_screen_inches,
        "max_weight_kg": extraction.max_weight_kg,
        "min_battery_hours": extraction.min_battery_hours,
        "required_os": extraction.required_os,
        "dedicated_gpu_required": extraction.dedicated_gpu_required,
        "touchscreen_required": extraction.touchscreen_required,
        "preferred_brands": extraction.preferred_brands,
        "excluded_brands": extraction.excluded_brands,
    }
    if extraction.budget_max is not None:
        data["budget_max"] = Money(
            amount=extraction.budget_max.amount, currency=extraction.budget_max.currency
        )
    if extraction.budget_min is not None:
        data["budget_min"] = Money(
            amount=extraction.budget_min.amount, currency=extraction.budget_min.currency
        )

    if previous is not None:
        merged = previous.model_dump()
        for key, value in data.items():
            if _is_unset(key, value) and not _is_unset(key, merged.get(key)):
                continue  # a default must never erase something already known
            if value is None:
                continue
            merged[key] = value
        merged["mandatory_fields"] = sorted(
            set(previous.mandatory_fields) | set(extraction.mandatory_fields)
        )
        data = merged
    else:
        data["mandatory_fields"] = extraction.mandatory_fields

    # Drop mandatory markers whose field ended up unset — the domain model
    # rejects a mandatory constraint with no value.
    candidate = LaptopRequirements.model_validate({**data, "mandatory_fields": []})
    valid_mandatory = [
        field
        for field in data.get("mandatory_fields", [])
        if getattr(candidate, field, None) not in (None, [], False)
    ]
    return candidate.model_copy(update={"mandatory_fields": sorted(set(valid_mandatory))})


def _to_profile(extraction: Any, previous: PurchaseProfile | None) -> PurchaseProfile:
    """Build the eligibility profile. Bank names only — never card numbers."""
    banks = list(extraction.eligible_banks)
    exchange = extraction.has_exchange_device
    emi = extraction.wants_no_cost_emi
    if previous is not None:
        banks = list({*previous.eligible_banks, *banks})
        exchange = exchange or previous.has_exchange_device
        emi = emi or previous.wants_no_cost_emi
    return PurchaseProfile(
        eligible_banks=banks, has_exchange_device=exchange, wants_no_cost_emi=emi
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _render_recommendation(
    recommendation: Recommendation, state: LaptopAgentState
) -> str:
    """Render the answer. Every figure comes from the validated recommendation."""
    lines = [
        f"**{recommendation.title}**",
        f"Marketplace: {recommendation.marketplace.value.title()}",
        "",
        f"Price: {recommendation.effective_price}",
    ]
    if recommendation.upfront_savings.amount > 0:
        lines.append(
            f"(list {recommendation.listed_price}, "
            f"{recommendation.upfront_savings} applied at checkout)"
        )
    if recommendation.cashback_value.amount > 0:
        lines.append(
            f"Cashback: {recommendation.cashback_value} — returned after purchase, "
            "not deducted at checkout."
        )
    if recommendation.unmet_conditional_offers:
        lines.append(
            f"{len(recommendation.unmet_conditional_offers)} further offer(s) exist "
            "but require conditions you have not confirmed, so they are excluded "
            "from the price above."
        )
    lines += ["", recommendation.rationale]

    if recommendation.trade_offs:
        lines += ["", "**Trade-offs**"]
        lines += [
            f"- {item.dimension}: {item.detail}" for item in recommendation.trade_offs
        ]
    if recommendation.runner_ups:
        lines += ["", "**Also considered**"]
        lines += [
            f"- {runner.title} ({runner.marketplace.value}) — {runner.effective_price}"
            + (f": {runner.why_not}" if runner.why_not else "")
            for runner in recommendation.runner_ups
        ]

    flags = state.get("injection_flags", [])
    if flags:
        lines += [
            "",
            f"Note: {len(flags)} listing(s) contained seller text attempting to "
            "influence this recommendation. Those listings were disqualified from "
            "being recommended, and their text was never treated as an instruction.",
        ]
    lines += ["", f"Link: {recommendation.url}"]
    return "\n".join(lines)
