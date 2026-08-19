"""Secret and PII redaction.

One redactor serves three consumers, so a pattern added here is applied
everywhere at once:

* structured logging (:mod:`laptop_agent.security.logging`)
* LangSmith trace payloads (``Client(hide_inputs=..., hide_outputs=...)``)
* free-text captured from the user before it is stored on a session

Redaction is deterministic regex work. It is a safety net, not a licence to log
secrets — the code paths deliberately never pass credentials into log calls.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

MASK: Final = "[REDACTED]"
_CARD_MASK: Final = "[REDACTED_CARD]"

#: Keys whose value is dropped entirely, regardless of what it looks like.
SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "anthropic_api_key",
        "langsmith_api_key",
        "openrouter_api_key",
        "serpapi_key",
        "authorization",
        "auth",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "bearer",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "card_number",
        "cardnumber",
        "cvv",
        "cvc",
        "pin",
        "otp",
        "account_number",
        "upi_id",
        "pan",
        "aadhaar",
        "ssn",
        "cookie",
        "set-cookie",
        "x-api-key",
    }
)

# Ordered: more specific patterns first so a generic rule cannot pre-empt them.
_PATTERNS: Final[tuple[tuple[str, re.Pattern[str], str], ...]] = (
    # Anthropic keys
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"), MASK),
    # OpenRouter keys — matched before the generic sk- rule, which would
    # otherwise stop at the first hyphen and leave most of the key visible.
    ("openrouter_key", re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"), MASK),
    # OpenAI-style keys (may appear in pasted user text)
    ("generic_sk_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), MASK),
    # LangSmith keys.
    #
    # No trailing \b, and underscores are part of the token class: a real key is
    # lsv2_pt_<32 hex>_<10 hex>, and "_" is a word character, so a trailing \b
    # never matches after the first segment. The original pattern left the whole
    # key visible.
    ("langsmith_key", re.compile(r"lsv2_(?:pt|sk)_[A-Za-z0-9_]{16,}"), MASK),
    # AWS access key id / secret
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), MASK),
    (
        "aws_secret_access_key",
        re.compile(
            r"(?i)\baws_secret_access_key\b\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
        ),
        MASK,
    ),
    # JWTs
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        MASK,
    ),
    # Authorization headers
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-/+=]{12,}"), f"Bearer {MASK}"),
    # key=value / key: value assignments for sensitive names.
    #
    # The name may carry a prefix or a suffix — LANGSMITH_API_KEY, X-Api-Key,
    # AWS_SECRET_ACCESS_KEY (sensitive word in the middle), db_password_hint.
    # "_" is a word character, so a bare \b before "api_key" does not match
    # inside "LANGSMITH_API_KEY"; the leading [\w.-]* is what makes prefixed
    # environment-variable names match.
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b([\w.-]*(?:api[_-]?key|secret|password|passwd|token|cvv|cvc|otp|pin)"
            r"[\w.-]*)\s*[:=]\s*['\"]?[^\s'\",;]{3,}['\"]?"
        ),
        r"\1=" + MASK,
    ),
    # Indian PAN / Aadhaar
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), MASK),
    ("aadhaar", re.compile(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b"), MASK),
    # UPI handle
    ("upi", re.compile(r"\b[\w.\-]{3,}@(?:okhdfcbank|okicici|oksbi|okaxis|upi|paytm|ybl)\b"), MASK),
    # Email
    ("email", re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]{2,}\b"), MASK),
    # Phone (India-leaning, 10 digits with optional +91)
    ("phone", re.compile(r"(?:(?<=\D)|^)(?:\+?91[\s-]?)?[6-9]\d{9}(?=\D|$)"), MASK),
    # CVV stated in prose
    ("cvv_prose", re.compile(r"(?i)\bcvv\b\D{0,10}\b\d{3,4}\b"), f"cvv {MASK}"),
)

#: 13-19 digit runs with optional separators — candidate card numbers.
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
#: "card ending in 1234" / "card ends with 1234" — keep the phrasing, drop digits.
_CARD_TAIL = re.compile(
    r"(?i)\b(card\s+(?:no\.?|number\s+)?(?:ending|ends)(?:\s+(?:in|with))?)\s*[:#]?\s*(\d{4})\b"
)
#: Bare 4-digit "last four" after a bank/card word.
_CARD_LAST4 = re.compile(r"(?i)\b(last\s*(?:4|four)\s*digits?)\D{0,8}(\d{4})\b")


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — keeps ordinary long numbers from being masked as cards."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


class SecretRedactor:
    """Redacts secrets and PII from strings and nested structures."""

    def __init__(self, *, extra_literals: Sequence[str] = ()) -> None:
        # Literal values (e.g. the configured API keys) are masked even if they
        # do not match a pattern.
        self._literals = tuple(
            literal for literal in extra_literals if literal and len(literal) >= 8
        )

    # ----- strings -----

    def redact_text(self, text: str) -> str:
        if not text:
            return text
        redacted = text
        for literal in self._literals:
            redacted = redacted.replace(literal, MASK)
        redacted = self._redact_cards(redacted)
        for _name, pattern, replacement in _PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    def _redact_cards(self, text: str) -> str:
        """Mask full card numbers and the trailing 4 digits people volunteer."""
        text = _CARD_TAIL.sub(rf"\1 {_CARD_MASK}", text)
        text = _CARD_LAST4.sub(rf"\1 {_CARD_MASK}", text)

        def _mask_if_card(match: re.Match[str]) -> str:
            digits = re.sub(r"[ -]", "", match.group(0))
            if 13 <= len(digits) <= 19 and _luhn_ok(digits):
                return _CARD_MASK
            return match.group(0)

        return _CARD_CANDIDATE.sub(_mask_if_card, text)

    # ----- structures -----

    def redact(self, value: Any, *, _depth: int = 0) -> Any:
        """Recursively redact a JSON-like structure.

        Depth is bounded so a pathological payload cannot cause runaway
        recursion inside a logging or tracing hook.
        """
        if _depth > 12:
            return MASK
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                if key_str.strip().lower().replace("-", "_") in SENSITIVE_KEYS:
                    result[key_str] = MASK
                else:
                    result[key_str] = self.redact(item, _depth=_depth + 1)
            return result
        if isinstance(value, (list, tuple, set)):
            return [self.redact(item, _depth=_depth + 1) for item in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        # Unknown object: stringify through the text path rather than trusting
        # its __repr__ to be secret-free.
        return self.redact_text(repr(value))

    # ----- LangSmith hooks -----

    def hide_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.redact(inputs)  # type: ignore[return-value]

    def hide_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        return self.redact(outputs)  # type: ignore[return-value]


def build_redactor() -> SecretRedactor:
    """Redactor seeded with the process's own configured secrets."""
    from ..config import get_settings

    settings = get_settings()
    literals = [
        secret.get_secret_value()
        for secret in (
            settings.anthropic_api_key,
            settings.langsmith_api_key,
            settings.openrouter_api_key,
            settings.serpapi_key,
        )
        if secret is not None
    ]
    return SecretRedactor(extra_literals=literals)
