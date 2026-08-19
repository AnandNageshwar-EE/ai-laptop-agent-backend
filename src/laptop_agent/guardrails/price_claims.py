"""Screening LLM prose for invented monetary claims.

The prompt tells the model not to state figures, and the schema keeps numbers
out of structured fields — but "the prompt said not to" is not a guardrail. This
module is the enforcement: any monetary figure or percentage appearing in
model-authored prose must match a value the deterministic pricing code actually
computed, or it is removed.

This closes the specific failure where a model helpfully writes
"saving you ₹12,000" from a discount it inferred rather than one that exists.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ..domain.money import Money

#: Currency-marked amounts: ₹79,999 / Rs. 79999 / INR 79,999.00 / $1,299
_CURRENCY_AMOUNT = re.compile(
    # No \b after "rs\.?" — with the dot present the boundary sits between "."
    # and a space, both non-word characters, so it can never match and "Rs. 48,999"
    # slipped through entirely. The leading \b is what prevents matching inside a
    # longer word.
    r"(?i)(?:₹|\brs\.?|\binr\b|\$|\busd\b)\s*([\d,]+(?:\.\d{1,2})?)"
    r"|([\d,]+(?:\.\d{1,2})?)\s*(?:₹|\brs\.?|\binr\b|\brupees?\b|\bdollars?\b)"
)

#: Indian-notation amounts: "80k", "1.5 lakh", "1,20,000"
_SCALED_AMOUNT = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*(k|lakh|lac|crore)\b")

#: Percentages: "20% off", "saves 15 percent".
#:
#: The word boundary belongs *after* "percent" only. A trailing \b following "%"
#: can never match — "%" is a non-word character and so is the space after it —
#: which meant "Save 20% off" passed through this screen entirely unstripped
#: while "15 percent" was caught.
_PERCENTAGE = re.compile(r"(?i)\b(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)")

#: Percentages that are display specifications, not discounts. A colour-gamut
#: figure is a genuine product fact, so stripping it would degrade a correct
#: explanation rather than protect anyone.
_PERCENT_SPEC = re.compile(
    r"(?i)\b\d{1,3}(?:\.\d+)?\s*%\s*"
    r"(?:s\s?rgb|dci[- ]?p3|adobe\s*rgb|ntsc|rec\.?\s*709|"
    r"screen[- ]to[- ]body|brightness|colou?r\s+(?:gamut|accuracy))"
)

#: Figures that are specifications, not prices. Matched first and exempted, so
#: "16GB RAM" and "14 inch" are never mistaken for monetary claims.
_SPEC_CONTEXT = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*"
    r"(?:gb|tb|mb|ghz|mhz|hz|inch(?:es)?|\"|cm|kg|g\b|wh|w\b|hours?|hrs?|"
    r"cores?|threads?|nits|ppi|mp|years?|months?|rpm)"
)

_SCALE = {"k": Decimal("1000"), "lakh": Decimal("100000"), "lac": Decimal("100000"),
          "crore": Decimal("10000000")}


class PriceClaimReport:
    """Outcome of screening one piece of prose."""

    def __init__(self, text: str, removed: list[str]) -> None:
        self.text = text
        self.removed = removed

    @property
    def had_unauthorised_claims(self) -> bool:
        return bool(self.removed)


def _parse(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def screen_price_claims(text: str, allowed: list[Money]) -> PriceClaimReport:
    """Remove monetary figures from ``text`` that are not in ``allowed``.

    Rather than rejecting the whole explanation — which would fail a request over
    a cosmetic slip — the offending clause is replaced with neutral wording and
    the incident is reported for the audit trail.
    """
    if not text:
        return PriceClaimReport(text, [])

    permitted: set[Decimal] = set()
    for money in allowed:
        permitted.add(money.amount)
        # Accept the rounded forms a person would naturally write.
        permitted.add(money.amount.quantize(Decimal("1")))
        permitted.add((money.amount / Decimal("1000")).quantize(Decimal("0.1")))
        permitted.add((money.amount / Decimal("100000")).quantize(Decimal("0.1")))

    removed: list[str] = []
    spec_spans = [match.span() for match in _SPEC_CONTEXT.finditer(text)]
    spec_spans += [match.span() for match in _PERCENT_SPEC.finditer(text)]

    def _in_spec_context(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in spec_spans)

    def _replace_currency(match: re.Match[str]) -> str:
        if _in_spec_context(*match.span()):
            return match.group(0)
        raw = match.group(1) or match.group(2) or ""
        value = _parse(raw)
        if value is not None and value in permitted:
            return match.group(0)
        removed.append(match.group(0).strip())
        return "the listed price"

    def _replace_scaled(match: re.Match[str]) -> str:
        if _in_spec_context(*match.span()):
            return match.group(0)
        value = _parse(match.group(1))
        unit = match.group(2).lower()
        if value is None:
            return match.group(0)
        absolute = value * _SCALE[unit]
        if absolute in permitted or value in permitted:
            return match.group(0)
        removed.append(match.group(0).strip())
        return "the listed price"

    def _replace_percent(match: re.Match[str]) -> str:
        """Remove a discount percentage entirely.

        Substituting a phrase produced sentences like "A a discount discount
        applies." Deleting the figure and letting the whitespace collapse leaves
        "A discount applies." — which is both accurate and readable.
        """
        if _in_spec_context(*match.span()):
            return match.group(0)
        removed.append(match.group(0).strip())
        return ""

    screened = _CURRENCY_AMOUNT.sub(_replace_currency, text)
    screened = _SCALED_AMOUNT.sub(_replace_scaled, screened)
    screened = _PERCENTAGE.sub(_replace_percent, screened)
    screened = re.sub(r"\s{2,}", " ", screened).strip()

    return PriceClaimReport(screened, removed)
