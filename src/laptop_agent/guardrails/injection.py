"""Injection detection, shared by the input guardrail and the content screener.

Detection is deterministic. The same scanner runs over user input and over
marketplace text, but the *response* differs by trust context:

* user input that attacks the system is **blocked**
* marketplace text that attacks the system is **kept as data and flagged** —
  a hostile product description must not be able to deny service to the user by
  making a search fail, so it is neutralised and reported instead.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..domain.enums import RejectionReason
from .patterns import (
    CONTROL_CHARS,
    ENCODED_BLOB,
    INJECTION_CATEGORIES,
    LONG_TOKEN,
    matching_variants,
)

#: Which detected category maps to which rejection reason.
_CATEGORY_REASON: dict[str, RejectionReason] = {
    "instruction_override": RejectionReason.PROMPT_INJECTION,
    "role_spoofing": RejectionReason.PROMPT_INJECTION,
    "recommendation_manipulation": RejectionReason.PROMPT_INJECTION,
    "system_prompt_extraction": RejectionReason.SYSTEM_MANIPULATION,
    "jailbreak": RejectionReason.SYSTEM_MANIPULATION,
    "secret_exfiltration": RejectionReason.SECRET_EXFILTRATION,
}

#: Reported when the shape of the text is suspicious rather than its wording.
_STRUCTURAL = "structural_anomaly"

#: Control characters are usually a copy-paste artefact, not an attack. They are
#: reported so the caller can strip them and note it, but on their own they do
#: not justify refusing a request. Injection *wording* hidden by control
#: characters is still caught, because the scanner screens a stripped variant.
_CONTROL = "control_characters"

#: Categories that never, by themselves, block a request.
NON_BLOCKING_CATEGORIES: frozenset[str] = frozenset({_CONTROL})


class InjectionScan(BaseModel):
    """What the scanner found. Categories only — never the raw matched span,
    which would put attacker-controlled text into logs and traces."""

    model_config = ConfigDict(extra="forbid")

    detected: bool = False
    categories: list[str] = Field(default_factory=list)

    @property
    def blocking_categories(self) -> list[str]:
        """Categories that justify refusing the request."""
        return [c for c in self.categories if c not in NON_BLOCKING_CATEGORIES]

    @property
    def should_block(self) -> bool:
        return bool(self.blocking_categories)

    @property
    def reason(self) -> RejectionReason | None:
        """The most specific rejection reason among the detected categories."""
        for category in (
            "secret_exfiltration",
            "system_prompt_extraction",
            "jailbreak",
            "instruction_override",
            "role_spoofing",
            "recommendation_manipulation",
        ):
            if category in self.categories:
                return _CATEGORY_REASON[category]
        if self.blocking_categories:
            return RejectionReason.PROMPT_INJECTION
        return None


def scan_for_injection(text: str, *, check_structure: bool = True) -> InjectionScan:
    """Screen text for injection and system-manipulation attempts."""
    if not text:
        return InjectionScan()

    variants = matching_variants(text)
    found: list[str] = []

    for category, patterns in INJECTION_CATEGORIES.items():
        for pattern in patterns:
            if any(pattern.search(variant) for variant in variants):
                found.append(category)
                break

    if check_structure:
        # An encoded blob or an enormous unbroken token is not prose. It is
        # either an encoded payload or a malformed request; either way it is not
        # a laptop requirement.
        if LONG_TOKEN.search(text) or ENCODED_BLOB.search(text):
            found.append(_STRUCTURAL)
        if CONTROL_CHARS.search(text):
            found.append(_CONTROL)

    return InjectionScan(detected=bool(found), categories=sorted(set(found)))


def strip_control_characters(text: str) -> tuple[str, bool]:
    """Remove control characters. Returns the text and whether any were removed."""
    cleaned = CONTROL_CHARS.sub("", text)
    return cleaned, cleaned != text
