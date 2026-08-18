"""Public agent entry point.

Owns the process-lifetime collaborators (prompt provider, marketplace registry,
product cache, tracing) and creates the per-run ones (metrics, nodes, graph).

Threading note: the shared objects are the cache, the registry and the prompt
provider — all internally synchronised or immutable. Per-run mutable state lives
on :class:`~laptop_agent.observability.metrics.RunMetrics` and
:class:`~laptop_agent.graph.nodes.AgentNodes`, both created fresh per request, so
concurrent sessions cannot interfere.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditSink, StructuredLogAuditSink
from .cache import InMemoryCacheProvider
from .config import Settings, get_settings
from .domain.recommendation import Recommendation
from .graph.builder import build_graph
from .graph.nodes import AgentNodes
from .graph.state import initial_state
from .guardrails.scope_guardrail import ConversationStage
from .llm.facade import Reasoner
from .marketplace.registry import build_registry
from .observability.metrics import RunMetrics
from .observability.tracing import TracingSetup, marketplace_tag
from .prompts.provider import PromptTask, get_prompt_provider
from .security.logging import configure_logging, get_logger
from .session import InMemorySessionStore, SessionState, SessionStore, new_session_id

_logger = get_logger("laptop_agent.agent")


class AgentReply(BaseModel):
    """What the API returns. No internal detail, no raw provider payloads."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    response_text: str
    #: True when the request was refused by a guardrail.
    blocked: bool = False
    block_reason: str | None = None
    awaiting_clarification: bool = False
    clarification_question: str = ""
    recommendation: Recommendation | None = None
    trade_off_required: bool = False
    warnings: list[str] = Field(default_factory=list)
    #: Non-sensitive run facts, also attached to the trace.
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class LaptopAgent:
    """The application. Construct once, call :meth:`run` per user turn."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        sessions: SessionStore | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self.settings = get_settings() if settings is None else settings
        configure_logging()
        self.prompts = get_prompt_provider()
        self.product_cache = InMemoryCacheProvider()
        self.registry = build_registry(cache=self.product_cache, settings=self.settings)
        self.tracing = TracingSetup(settings=self.settings)
        # Explicit None checks, not `or`: an empty InMemorySessionStore has
        # len() == 0 and is therefore falsy, so `sessions or default` would
        # silently discard an injected store on its first use.
        self.audit = StructuredLogAuditSink() if audit is None else audit
        self.sessions = (
            InMemorySessionStore(
                ttl_seconds=self.settings.session_ttl_seconds,
                max_sessions=self.settings.max_sessions,
            )
            if sessions is None
            else sessions
        )
        # Built once: constructing the model per request would waste connections
        # and, on the live path, is unnecessary.
        self.reasoner = Reasoner(settings=self.settings, prompts=self.prompts)

    # ------------------------------------------------------------------

    def run(self, *, message: str, session_id: str | None = None) -> AgentReply:
        """Process one user turn."""
        resolved_session = session_id or new_session_id()
        session = self.sessions.get(resolved_session) or SessionState(
            session_id=resolved_session
        )

        stage = (
            ConversationStage.AWAITING_CLARIFICATION
            if session.pending_clarification
            else (
                ConversationStage.FOLLOW_UP
                if session.turn_count > 0
                else ConversationStage.OPENING
            )
        )

        metrics = RunMetrics(session_id=resolved_session)
        nodes = AgentNodes(
            metrics=metrics,
            settings=self.settings,
            registry=self.registry,
            reasoner=self.reasoner,
            audit=self.audit,
            stage=stage,
        )
        graph = build_graph(nodes, settings=self.settings)

        state = initial_state(
            session_id=resolved_session,
            user_request=message,
            requirements=session.requirements,
            purchase_profile=session.purchase_profile,
            turn_count=session.turn_count,
        )

        config = self.tracing.run_config(
            session_id=resolved_session,
            extra_metadata={
                **self.prompts.versions(PromptTask.REQUIREMENTS),
                "conversation_stage": stage.value,
                "turn": session.turn_count + 1,
                "marketplace_provider_count": len(self.registry),
            },
            extra_tags=[
                marketplace_tag(market.value) for market in self.registry.marketplaces
            ],
        )

        try:
            final = graph.invoke(state, config=config)
        except Exception as exc:
            _logger.error(
                "graph.run_failed",
                extra={"session_id": resolved_session, "error_type": type(exc).__name__},
                exc_info=True,
            )
            return AgentReply(
                session_id=resolved_session,
                response_text=(
                    "Something went wrong on my side while searching. Please try again."
                ),
                blocked=False,
                diagnostics={"error": type(exc).__name__},
            )

        self._persist_session(session, final)
        return self._to_reply(resolved_session, final, metrics, stage)

    # ------------------------------------------------------------------

    def _persist_session(self, session: SessionState, final: dict[str, Any]) -> None:
        """Store only requirements and eligibility. Never products or prices."""
        if final.get("blocked"):
            # A refused turn must not advance or mutate the session.
            return
        updated = session.model_copy(
            update={
                "requirements": final.get("requirements") or session.requirements,
                "purchase_profile": final.get("purchase_profile")
                or session.purchase_profile,
                "pending_clarification": (
                    final.get("clarification_question")
                    if final.get("needs_clarification")
                    else None
                ),
                "turn_count": session.turn_count + 1,
            }
        )
        self.sessions.put(updated)

    def _to_reply(
        self,
        session_id: str,
        final: dict[str, Any],
        metrics: RunMetrics,
        stage: ConversationStage,
    ) -> AgentReply:
        requirements = final.get("requirements")
        metrics.marketplaces_used = list(final.get("marketplaces_used", []))

        diagnostics = {
            **metrics.as_trace_metadata(),
            "user_requirement_category": (
                requirements.use_case.value if requirements is not None else "unknown"
            ),
            "clarification_required": bool(final.get("needs_clarification")),
            "trade_off_required": bool(final.get("trade_off_required")),
            "selected_candidate": (
                f"{final['recommendation'].marketplace.value}:"
                f"{final['recommendation'].product_id}"
                if final.get("recommendation") is not None
                else None
            ),
            "conversation_stage": stage.value,
            "prompt_versions": self.prompts.versions(PromptTask.RECOMMENDATION),
            "prompt_assembly_cache": self.prompts.assembly_stats(),
            "quarantined_products": [key for key, _ in final.get("quarantined_products", [])],
            "quarantined_offers": [key for key, _ in final.get("quarantined_offers", [])],
        }

        return AgentReply(
            session_id=session_id,
            response_text=final.get("response_text", ""),
            blocked=bool(final.get("blocked")),
            block_reason=final.get("block_reason"),
            awaiting_clarification=bool(final.get("needs_clarification")),
            clarification_question=final.get("clarification_question", ""),
            recommendation=final.get("recommendation"),
            trade_off_required=bool(final.get("trade_off_required")),
            warnings=list(final.get("warnings", [])),
            diagnostics=diagnostics,
        )

    def shutdown(self) -> None:
        self.tracing.flush()


_agent: LaptopAgent | None = None


def get_agent() -> LaptopAgent:
    global _agent
    if _agent is None:
        _agent = LaptopAgent()
    return _agent


def reset_agent() -> None:
    """Test helper."""
    global _agent
    _agent = None
