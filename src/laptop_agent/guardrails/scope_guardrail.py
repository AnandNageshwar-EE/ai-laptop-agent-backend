"""Scope guardrail — is this a laptop shopping request at all?

Two distinct decisions, kept separate because they have different failure costs:

**Disallowed topic** — malware, intrusion, fraud. Deterministic deny. A false
negative here is a security incident, so the patterns are broad and the check
runs on every turn regardless of conversation state.

**Off-topic** — a request that is harmless but not about laptops. This one needs
care: a legitimate answer to a clarifying question ("yes", "80k", "hdfc") shares
no vocabulary with the laptop lexicon, so a naive relevance check would reject
the user's own follow-up. Relevance is therefore evaluated against the
conversation stage, and terse answers to a pending question are allowed through.

An LLM classifier may act as a tie-breaker on the genuinely ambiguous middle,
but it is advisory only: it can never overturn a deterministic deny, and when it
is unavailable the deterministic verdict stands.
"""

from __future__ import annotations

from enum import StrEnum

from ..domain.enums import RejectionReason
from .patterns import (
    DISALLOWED_TOPICS,
    ON_TOPIC_TERMS,
    is_terse_answer,
    matching_variants,
    tokenise,
)
from .result import GuardrailResult


class ConversationStage(StrEnum):
    #: First turn of a session — the request must stand on its own.
    OPENING = "opening"
    #: The agent asked a question and is awaiting an answer.
    AWAITING_CLARIFICATION = "awaiting_clarification"
    #: Mid-session follow-up after a recommendation was given.
    FOLLOW_UP = "follow_up"


class ScopeVerdict(StrEnum):
    IN_SCOPE = "in_scope"
    OFF_TOPIC = "off_topic"
    DISALLOWED = "disallowed"
    #: Deterministically unclear — eligible for an advisory classifier.
    UNCLEAR = "unclear"


class ScopeGuardrail:
    """Keeps the agent inside the laptop/product shopping domain."""

    #: At least this many on-topic terms makes a request unambiguously in scope.
    _STRONG_SIGNAL = 2

    def check(
        self,
        text: str,
        *,
        stage: ConversationStage = ConversationStage.OPENING,
    ) -> GuardrailResult[str]:
        verdict, detail = self.classify(text, stage=stage)

        if verdict is ScopeVerdict.DISALLOWED:
            return GuardrailResult.block(
                RejectionReason.DISALLOWED_TOPIC,
                internal_detail={"check": "disallowed_topic", **detail},
            )
        if verdict is ScopeVerdict.OFF_TOPIC:
            return GuardrailResult.block(
                RejectionReason.OUT_OF_SCOPE,
                internal_detail={"check": "off_topic", **detail},
            )
        # IN_SCOPE and UNCLEAR both proceed. UNCLEAR is annotated so the caller
        # may consult the advisory classifier before spending a search.
        return GuardrailResult.allow(
            text, notes=["scope_unclear"] if verdict is ScopeVerdict.UNCLEAR else []
        )

    def classify(
        self,
        text: str,
        *,
        stage: ConversationStage = ConversationStage.OPENING,
    ) -> tuple[ScopeVerdict, dict[str, object]]:
        """Deterministic scope classification. No model call."""
        variants = matching_variants(text)

        # ---- disallowed topics: checked first, and on every stage ----
        for topic, patterns in DISALLOWED_TOPICS.items():
            for pattern in patterns:
                if any(pattern.search(variant) for variant in variants):
                    return ScopeVerdict.DISALLOWED, {"topic": topic}

        # ---- a terse answer to a pending question is in scope by context ----
        if stage is ConversationStage.AWAITING_CLARIFICATION and is_terse_answer(text):
            return ScopeVerdict.IN_SCOPE, {"reason": "terse_clarification_answer"}

        # ---- relevance ----
        matched = tokenise(variants[0]) & ON_TOPIC_TERMS
        if len(matched) >= self._STRONG_SIGNAL:
            return ScopeVerdict.IN_SCOPE, {"matched_terms": len(matched)}
        if matched:
            # One signal only. Enough to continue mid-conversation, not enough
            # to open a session with.
            if stage is not ConversationStage.OPENING:
                return ScopeVerdict.IN_SCOPE, {"matched_terms": 1}
            return ScopeVerdict.UNCLEAR, {"matched_terms": 1}
        if stage is ConversationStage.AWAITING_CLARIFICATION:
            # A longer free-text answer to our own question. Give it the benefit
            # of the doubt — we asked, after all.
            return ScopeVerdict.UNCLEAR, {"reason": "answer_to_our_question"}
        return ScopeVerdict.OFF_TOPIC, {"matched_terms": 0}
