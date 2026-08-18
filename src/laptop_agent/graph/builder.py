"""Graph construction.

Flow:

    input_guardrail
        ├─ blocked ────────────────────────────────────────► END
        └─ requirements_analysis
               └─ clarification_decision
                      ├─ needs clarification ──────────────► END (awaits answer)
                      └─ search_planning
                             ├─► amazon_search   ─┐  (concurrent)
                             └─► flipkart_search ─┘
                                    └─ offer_analysis
                                          └─ pricing_calculation
                                                └─ product_ranking
                                                      ├─ no candidates ──► no_results ──► END
                                                      └─ recommendation_generation
                                                            └─ recommendation_validation
                                                                  ├─ valid ──────────► END
                                                                  ├─ retry left ─────► product_ranking
                                                                  └─ exhausted ──────► validation_failed ──► END

The repair loop is the mechanism required by spec section 1.7: a failed
validation routes *back through the graph* with the offending candidate
excluded, rather than letting the model argue with the validator.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config import Settings, get_settings
from .nodes import AgentNodes
from .state import LaptopAgentState


def route_after_input(state: LaptopAgentState) -> str:
    return "blocked" if state.get("blocked") else "continue"


def route_after_clarification(state: LaptopAgentState) -> str:
    return "clarify" if state.get("needs_clarification") else "search"


def route_after_requirements(state: LaptopAgentState) -> str:
    # A structured-output failure sets `blocked` with a graceful message.
    return "blocked" if state.get("blocked") else "continue"


def route_after_ranking(state: LaptopAgentState) -> str:
    return "recommend" if state.get("ranked_products") else "no_results"


def make_route_after_validation(max_attempts: int) -> Any:
    def route_after_validation(state: LaptopAgentState) -> str:
        if not state.get("validation_failures"):
            return "done"
        if state.get("repair_attempts", 0) >= max_attempts:
            return "exhausted"
        # Anything still rankable after the exclusion is worth another pass.
        remaining = [
            score
            for score in state.get("ranked_products", [])
            if f"{score.marketplace.value}:{score.product_id}"
            not in set(state.get("excluded_candidates", []))
        ]
        return "retry" if remaining else "exhausted"

    return route_after_validation


def build_graph(nodes: AgentNodes, *, settings: Settings | None = None) -> Any:
    """Compile the workflow.

    No checkpointer is attached: the backend is stateless with respect to product
    data, and the small amount of cross-turn state (requirements, eligibility)
    lives in the session store. Attaching one is a one-line change if durable
    resume is ever required.
    """
    resolved = get_settings() if settings is None else settings
    graph: StateGraph = StateGraph(LaptopAgentState)

    graph.add_node("input_guardrail", nodes.input_guardrail_node)
    graph.add_node("requirements_analysis", nodes.requirements_analysis_node)
    graph.add_node("clarification_decision", nodes.clarification_decision_node)
    graph.add_node("search_planning", nodes.search_planning_node)
    graph.add_node("amazon_search", nodes.amazon_search_node)
    graph.add_node("flipkart_search", nodes.flipkart_search_node)
    graph.add_node("offer_analysis", nodes.offer_analysis_node)
    graph.add_node("pricing_calculation", nodes.pricing_calculation_node)
    graph.add_node("product_ranking", nodes.product_ranking_node)
    graph.add_node("recommendation_generation", nodes.recommendation_generation_node)
    graph.add_node("recommendation_validation", nodes.recommendation_validation_node)
    graph.add_node("no_results", nodes.no_results_node)
    graph.add_node("validation_failed", nodes.validation_failed_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input,
        {"blocked": END, "continue": "requirements_analysis"},
    )
    graph.add_conditional_edges(
        "requirements_analysis",
        route_after_requirements,
        {"blocked": END, "continue": "clarification_decision"},
    )
    graph.add_conditional_edges(
        "clarification_decision",
        route_after_clarification,
        {"clarify": END, "search": "search_planning"},
    )

    # Fan out to both marketplaces, then join. `products` uses an additive
    # reducer so both branches can write it concurrently.
    graph.add_edge("search_planning", "amazon_search")
    graph.add_edge("search_planning", "flipkart_search")
    graph.add_edge("amazon_search", "offer_analysis")
    graph.add_edge("flipkart_search", "offer_analysis")

    graph.add_edge("offer_analysis", "pricing_calculation")
    graph.add_edge("pricing_calculation", "product_ranking")
    graph.add_conditional_edges(
        "product_ranking",
        route_after_ranking,
        {"recommend": "recommendation_generation", "no_results": "no_results"},
    )
    graph.add_edge("recommendation_generation", "recommendation_validation")
    graph.add_conditional_edges(
        "recommendation_validation",
        make_route_after_validation(resolved.recommendation_max_repair_attempts),
        {
            "done": END,
            "retry": "product_ranking",
            "exhausted": "validation_failed",
        },
    )
    graph.add_edge("no_results", END)
    graph.add_edge("validation_failed", END)

    return graph.compile()
