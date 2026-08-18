"""Audit trail.

Guardrail decisions, price validations and recommendation verdicts are
security-relevant events. They are emitted as an append-only stream of
structured, redacted log records — the natural shape for this data and already
collectable by any log pipeline. LangSmith holds the correlated run traces.

:class:`AuditSink` is a Protocol so a durable sink can be added if immutable,
queryable retention is ever required.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..security.logging import get_logger

_logger = get_logger("laptop_agent.audit")


class AuditEvent(StrEnum):
    INPUT_BLOCKED = "input_blocked"
    SCOPE_REJECTED = "scope_rejected"
    INJECTION_DETECTED = "injection_detected"
    TOOL_ARGS_REJECTED = "tool_args_rejected"
    PRODUCT_QUARANTINED = "product_quarantined"
    OFFER_QUARANTINED = "offer_quarantined"
    PRICE_INVALID = "price_invalid"
    LLM_OUTPUT_INVALID = "llm_output_invalid"
    PRICE_CLAIM_STRIPPED = "price_claim_stripped"
    RECOMMENDATION_REJECTED = "recommendation_rejected"
    RECOMMENDATION_APPROVED = "recommendation_approved"


class AuditRecord(BaseModel):
    """One audit event. Field values are redacted by the log formatter."""

    model_config = ConfigDict(extra="forbid")

    event: AuditEvent
    session_id: str
    node: str = ""
    reason: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class AuditSink(Protocol):
    def record(self, record: AuditRecord) -> None: ...


class StructuredLogAuditSink:
    """Writes audit records to the redacting structured logger."""

    #: Events that indicate an attack or a data-integrity failure.
    _WARN_EVENTS = frozenset(
        {
            AuditEvent.INPUT_BLOCKED,
            AuditEvent.INJECTION_DETECTED,
            AuditEvent.TOOL_ARGS_REJECTED,
            AuditEvent.PRODUCT_QUARANTINED,
            AuditEvent.OFFER_QUARANTINED,
            AuditEvent.PRICE_INVALID,
            AuditEvent.RECOMMENDATION_REJECTED,
        }
    )

    def record(self, record: AuditRecord) -> None:
        level = 30 if record.event in self._WARN_EVENTS else 20  # WARNING / INFO
        _logger.log(
            level,
            "audit.%s",
            record.event.value,
            extra={
                "audit_event": record.event.value,
                "session_id": record.session_id,
                "node": record.node,
                "reason": record.reason,
                "detail": record.detail,
            },
        )


class CollectingAuditSink:
    """Test double — keeps records in memory."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def record(self, record: AuditRecord) -> None:
        self.records.append(record)

    def events(self) -> list[AuditEvent]:
        return [r.event for r in self.records]
