"""Append-only audit trail for guardrail and validator decisions."""

from .sink import (
    AuditEvent,
    AuditRecord,
    AuditSink,
    CollectingAuditSink,
    StructuredLogAuditSink,
)

__all__ = [
    "AuditEvent",
    "AuditRecord",
    "AuditSink",
    "CollectingAuditSink",
    "StructuredLogAuditSink",
]
