"""Guardrail verdict types.

A guardrail never raises for an expected rejection and never returns a bare
bool. It returns a verdict carrying the sanitised value, the machine-readable
reason, the user-facing message, and the internal detail for the audit log.

The split between ``user_message`` and ``internal_detail`` is deliberate: the
detail may name the pattern that matched, which is exactly the implementation
information that must not be echoed to whoever probed for it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import RejectionReason

T = TypeVar("T")


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    #: Value was modified (sanitised/stripped) but the request continues.
    SANITISE = "sanitise"
    #: Request stops here; a safe canned response is returned.
    BLOCK = "block"


#: The canned responses.
#:
#: Two rules shape these.
#:
#: **The injection family must be word-for-word identical.** ``PROMPT_INJECTION``,
#: ``SYSTEM_MANIPULATION`` and ``SECRET_EXFILTRATION`` share one message, so
#: someone probing for the prompt learns nothing from which reply comes back. A
#: different wording per category would be a free oracle.
#:
#: **Nothing describes the filter.** No reply mentions a pattern, a rule, a
#: category, or that anything was detected — that is exactly the implementation
#: detail an attacker is fishing for. Each one refuses plainly, says what the
#: agent does handle, and invites a real request.

#: Shared by every reply to an attempted manipulation. Do not vary per category.
_MANIPULATION_REPLY = (
    "I can't help with that. I'm a laptop shopping assistant — I can find "
    "laptops, compare prices across Amazon and Flipkart, and work out what "
    "you'd actually pay after offers. What kind of laptop are you looking for?"
)

SAFE_RESPONSES: dict[RejectionReason, str] = {
    RejectionReason.EMPTY_INPUT: (
        "I didn't catch any laptop requirements there. Tell me what you need — a "
        "budget, what you'll mainly use it for, and any must-have specifications."
    ),
    RejectionReason.INPUT_TOO_LONG: (
        "That message is too long for me to work with. Could you describe your "
        "laptop requirements in a few sentences?"
    ),
    RejectionReason.MALFORMED_INPUT: (
        "I couldn't read that. Please describe your laptop requirements in plain "
        "text — for example, \"a laptop for coding under 80,000 with 16GB RAM\"."
    ),
    RejectionReason.PROMPT_INJECTION: _MANIPULATION_REPLY,
    RejectionReason.SYSTEM_MANIPULATION: _MANIPULATION_REPLY,
    RejectionReason.SECRET_EXFILTRATION: _MANIPULATION_REPLY,
    RejectionReason.DISALLOWED_TOPIC: (
        "I can't help with that. I only handle laptop shopping — finding models, "
        "comparing prices, and checking which offers you qualify for."
    ),
    RejectionReason.OUT_OF_SCOPE: (
        "That's outside what I can help with. I'm a laptop shopping assistant: "
        "tell me your budget and what you'll use the laptop for, and I'll compare "
        "options across Amazon and Flipkart."
    ),
}


class GuardrailResult(BaseModel, Generic[T]):
    """Outcome of one guardrail check."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    action: GuardrailAction
    value: T | None = None
    reason: RejectionReason | None = None
    #: Safe, user-facing text. Populated only when blocked.
    user_message: str = ""
    #: For the audit log and traces. May name matched rules. Never shown to users.
    internal_detail: dict[str, Any] = Field(default_factory=dict)
    #: Non-fatal observations (e.g. control characters stripped).
    notes: list[str] = Field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.action is not GuardrailAction.BLOCK

    @property
    def blocked(self) -> bool:
        return self.action is GuardrailAction.BLOCK

    @classmethod
    def allow(cls, value: T, *, notes: list[str] | None = None) -> GuardrailResult[T]:
        return cls(action=GuardrailAction.ALLOW, value=value, notes=notes or [])

    @classmethod
    def sanitised(
        cls,
        value: T,
        *,
        notes: list[str] | None = None,
        internal_detail: dict[str, Any] | None = None,
    ) -> GuardrailResult[T]:
        return cls(
            action=GuardrailAction.SANITISE,
            value=value,
            notes=notes or [],
            internal_detail=internal_detail or {},
        )

    @classmethod
    def block(
        cls,
        reason: RejectionReason,
        *,
        internal_detail: dict[str, Any] | None = None,
    ) -> GuardrailResult[T]:
        return cls(
            action=GuardrailAction.BLOCK,
            reason=reason,
            user_message=SAFE_RESPONSES[reason],
            internal_detail=internal_detail or {},
        )


class ValidationOutcome(BaseModel, Generic[T]):
    """Result of validating a batch of external items — kept vs quarantined."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    accepted: list[T] = Field(default_factory=list)
    #: (identifier, reason) for each rejected item. Never silently dropped.
    quarantined: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined)
