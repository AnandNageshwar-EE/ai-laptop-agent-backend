"""Input guardrail — the first thing a request meets, before the graph runs.

Order of checks is deliberate: cheap structural checks first, so a 5 MB payload
is rejected on length rather than after running a dozen regexes over it.

The guardrail returns a *sanitised* string. Downstream code uses that value, not
the original, so control characters and invisible formatting cannot reach the
model even when the request is allowed through.
"""

from __future__ import annotations

import unicodedata

from ..config import Settings, get_settings
from ..domain.enums import RejectionReason
from ..security.redaction import SecretRedactor
from .injection import scan_for_injection, strip_control_characters
from .patterns import INVISIBLE_CHARS, has_letters
from .result import GuardrailResult


class InputGuardrail:
    """Validates and sanitises raw user input."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self._settings = get_settings() if settings is None else settings
        self._redactor = redactor

    def check(self, raw: object) -> GuardrailResult[str]:
        """Validate one user turn.

        ``raw`` is typed ``object`` on purpose — this is a trust boundary, and a
        non-string arriving from a deserialiser is a malformed-input case, not a
        ``TypeError`` deep in the graph.
        """
        settings = self._settings

        # ---- type / shape ----
        if raw is None:
            return GuardrailResult.block(
                RejectionReason.EMPTY_INPUT, internal_detail={"check": "none_input"}
            )
        if not isinstance(raw, str):
            return GuardrailResult.block(
                RejectionReason.MALFORMED_INPUT,
                internal_detail={"check": "not_a_string", "type": type(raw).__name__},
            )

        # ---- size, before any expensive work ----
        if len(raw) > settings.max_input_chars:
            return GuardrailResult.block(
                RejectionReason.INPUT_TOO_LONG,
                internal_detail={
                    "check": "max_length",
                    "length": len(raw),
                    "limit": settings.max_input_chars,
                },
            )

        notes: list[str] = []

        # ---- character hygiene ----
        cleaned, had_control = strip_control_characters(raw)
        if had_control:
            notes.append("control_characters_removed")
        if INVISIBLE_CHARS.search(cleaned):
            cleaned = INVISIBLE_CHARS.sub("", cleaned)
            notes.append("invisible_characters_removed")
        cleaned = unicodedata.normalize("NFKC", cleaned).strip()

        # ---- emptiness, after cleaning ----
        if not cleaned:
            return GuardrailResult.block(
                RejectionReason.EMPTY_INPUT, internal_detail={"check": "empty_after_clean"}
            )
        if len(cleaned) < settings.min_input_chars:
            return GuardrailResult.block(
                RejectionReason.EMPTY_INPUT,
                internal_detail={"check": "min_length", "length": len(cleaned)},
            )
        if not has_letters(cleaned) and not cleaned.replace(" ", "").isdigit():
            # Neither prose nor a number — punctuation soup.
            return GuardrailResult.block(
                RejectionReason.MALFORMED_INPUT,
                internal_detail={"check": "no_alphanumeric_content"},
            )

        # ---- injection / manipulation ----
        # Scanned against the raw text: the scanner derives its own normalised
        # variants, so obfuscation via invisible or control characters is caught
        # here rather than depending on the cleaning above.
        scan = scan_for_injection(raw)
        if scan.should_block:
            reason = scan.reason or RejectionReason.PROMPT_INJECTION
            return GuardrailResult.block(
                reason,
                internal_detail={
                    "check": "injection_scan",
                    "categories": scan.blocking_categories,
                },
            )

        # ---- strip secrets the user volunteered ----
        # A card number in the request must not travel further. It is removed
        # here, not merely masked at the log boundary.
        if self._redactor is not None:
            redacted = self._redactor.redact_text(cleaned)
            if redacted != cleaned:
                notes.append("sensitive_values_removed")
                cleaned = redacted

        if notes:
            return GuardrailResult.sanitised(
                cleaned, notes=notes, internal_detail={"check": "sanitised"}
            )
        return GuardrailResult.allow(cleaned)
