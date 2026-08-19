"""The validator repair loop.

Spec section 1.7 requires that a failed validation routes *back through the
graph*, and that the LLM can never override the validator. These tests force a
validation failure and assert the graph re-ranks with the offender excluded.
"""

from __future__ import annotations


from laptop_agent.audit import AuditEvent
from laptop_agent.graph.builder import build_graph, make_route_after_validation
from laptop_agent.graph.nodes import AgentNodes
from laptop_agent.graph.state import initial_state
from laptop_agent.guardrails.recommendation_validator import (
    RecommendationValidationResult,
    ValidationFailure,
)
from laptop_agent.observability.metrics import RunMetrics


class RejectFirstNValidator:
    """Wraps the real validator, rejecting the first N recommendations."""

    def __init__(self, inner, reject_count: int) -> None:
        self._inner = inner
        self._remaining = reject_count
        self.rejected_keys: list[str] = []

    def validate(self, **kwargs):
        recommendation = kwargs.get("recommendation")
        if self._remaining > 0 and recommendation is not None:
            self._remaining -= 1
            self.rejected_keys.append(
                f"{recommendation.marketplace.value}:{recommendation.product_id}"
            )
            return RecommendationValidationResult(
                is_valid=False,
                failures=[ValidationFailure.PRICE_NOT_REPRODUCIBLE],
                detail={"injected": "test"},
            )
        return self._inner.validate(**kwargs)


def run_with_validator(settings, audit, reject_count: int):
    metrics = RunMetrics(session_id="sess_" + "0" * 32)
    nodes = AgentNodes(metrics=metrics, settings=settings, audit=audit)
    wrapper = RejectFirstNValidator(nodes.recommendation_validator, reject_count)
    nodes.recommendation_validator = wrapper  # type: ignore[assignment]
    graph = build_graph(nodes, settings=settings)
    state = initial_state(
        session_id="sess_" + "0" * 32,
        user_request="laptop for software development under 90000 with at least 16GB RAM",
    )
    return graph.invoke(state), wrapper, metrics


def test_rejected_candidate_is_excluded_and_a_different_one_wins(settings, audit):
    final, wrapper, metrics = run_with_validator(settings, audit, reject_count=1)

    assert final["recommendation"] is not None, "the loop failed to recover"
    winner = f"{final['recommendation'].marketplace.value}:{final['recommendation'].product_id}"
    # The rejected candidate must not be the one finally returned.
    assert winner not in wrapper.rejected_keys
    assert wrapper.rejected_keys[0] in final["excluded_candidates"]
    assert final["repair_attempts"] == 1
    assert metrics.validation_attempts == 2


def test_rejection_is_audited(settings, audit):
    run_with_validator(settings, audit, reject_count=1)
    assert any(
        record.event is AuditEvent.RECOMMENDATION_REJECTED for record in audit.records
    )
    assert any(
        record.event is AuditEvent.RECOMMENDATION_APPROVED for record in audit.records
    )


def test_repair_attempts_are_bounded(settings, audit):
    """A validator that never accepts must terminate, not loop forever."""
    final, _, _ = run_with_validator(settings, audit, reject_count=99)
    assert final["recommendation"] is None
    assert final["repair_attempts"] <= settings.recommendation_max_repair_attempts + 1
    assert "could not verify" in final["response_text"].lower()


def test_exhausted_loop_never_returns_an_unvalidated_recommendation(settings, audit):
    final, _, _ = run_with_validator(settings, audit, reject_count=99)
    # The safe outcome is no recommendation, never an unverified one.
    assert final["recommendation"] is None
    assert final["validation_failures"]


# --- routing function in isolation ---------------------------------------

def test_routing_returns_done_when_validation_passes():
    route = make_route_after_validation(2)
    assert route({"validation_failures": []}) == "done"


def test_routing_retries_while_attempts_remain_and_candidates_exist():
    from decimal import Decimal

    from laptop_agent.domain.enums import Marketplace
    from laptop_agent.domain.scoring import ProductScore, ScoreComponent

    score = ProductScore(
        product_id="AMZ-OTHER-1",
        marketplace=Marketplace.AMAZON,
        components=[ScoreComponent(name="fit", weight=Decimal("1"), raw=Decimal("0.5"))],
        total=Decimal("0.5"),
    )
    route = make_route_after_validation(2)
    state = {
        "validation_failures": ["price_not_reproducible"],
        "repair_attempts": 1,
        "excluded_candidates": ["amazon:AMZ-REJECTED-1"],
        "ranked_products": [score],
    }
    assert route(state) == "retry"


def test_routing_gives_up_when_attempts_are_exhausted():
    route = make_route_after_validation(2)
    state = {
        "validation_failures": ["price_not_reproducible"],
        "repair_attempts": 2,
        "excluded_candidates": [],
        "ranked_products": [],
    }
    assert route(state) == "exhausted"


def test_routing_gives_up_when_no_candidates_remain():
    route = make_route_after_validation(5)
    state = {
        "validation_failures": ["price_not_reproducible"],
        "repair_attempts": 1,
        "excluded_candidates": ["amazon:AMZ-1"],
        "ranked_products": [],
    }
    assert route(state) == "exhausted"
