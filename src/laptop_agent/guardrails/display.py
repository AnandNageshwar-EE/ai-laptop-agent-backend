"""Neutralising untrusted text for display.

Separate from :mod:`.untrusted`, which frames untrusted content for the *model*.
This module prepares it for a *human*.

A seller-authored title like ``"Budget Laptop 15 -- IGNORE PREVIOUS
INSTRUCTIONS"`` is data, and the pipeline is right not to crash on it. But
echoing it verbatim into the user's screen is its own defect: it hands the
attacker a channel to the user, and it makes the product look broken rather than
the listing look suspect.

Applied at the HTTP boundary, after the recommendation validator has compared
titles against provider data — so validation still works on the exact provider
bytes, and only presentation is cleaned.
"""

from __future__ import annotations

import re
import unicodedata

from .patterns import INJECTION_CATEGORIES

#: What a neutralised span becomes.
MARKER = "[removed]"

#: Runs of shouting, which is what most of these payloads look like.
_SHOUTING = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){2,}\b")


def neutralise_for_display(text: str) -> tuple[str, bool]:
    """Return ``(clean_text, was_modified)``.

    Instruction-like spans are replaced rather than the whole string dropped, so
    the legitimate part of a title still identifies the product.
    """
    if not text:
        return text, False

    # Sellers use mathematical-alphanumeric look-alikes to stand out in listings
    # and to slip past keyword filters: "𝗟𝗲𝗻𝗼𝘃𝗼" is not "Lenovo" to a matcher but
    # is to a reader. NFKC folds them to ASCII before anything else runs.
    cleaned = unicodedata.normalize("NFKC", text)
    for patterns in INJECTION_CATEGORIES.values():
        for pattern in patterns:
            cleaned = pattern.sub(MARKER, cleaned)

    # Collapse any leftover all-caps imperative, which reads as an instruction
    # to a human even when it matches no specific pattern.
    cleaned = _SHOUTING.sub(MARKER, cleaned)

    # Tidy the punctuation left behind by a removal.
    cleaned = re.sub(rf"[\s\-–—:;,]+{re.escape(MARKER)}", f" {MARKER}", cleaned)
    cleaned = re.sub(rf"(?:{re.escape(MARKER)}\s*)+", f"{MARKER} ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -–—:;,")

    return cleaned, cleaned != text
