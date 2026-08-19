"""Deterministic reasoning, used when no live model is configured.

This is not a mock. It implements the same three operations against the same
schemas using rules, so the full graph — guardrails, pricing, ranking,
validation, tracing — is exercisable in CI and in a demo with no API key and no
network. ``LLM_MODE=offline`` selects it; ``LLM_MODE=live`` uses the model.

It is intentionally conservative: where a rule cannot determine a field it leaves
it unset, exactly as the prompt instructs the model to.
"""

from __future__ import annotations

import re
from decimal import Decimal

from ..domain.enums import Bank, UseCase
from ..domain.product import ProductCandidate
from ..domain.requirements import LaptopRequirements
from .schemas import (
    ClarificationDecision,
    ExtractedBudget,
    RecommendationExplanation,
    RequirementExtraction,
    RunnerUpNote,
    SearchDecision,
    TradeOffOut,
)

# ---------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------

_USE_CASE_TERMS: tuple[tuple[UseCase, tuple[str, ...]], ...] = (
    (UseCase.GAMING, ("gaming", "game", "games", "fps", "esports", "valorant", "gta")),
    (UseCase.DATA_SCIENCE, ("data science", "machine learning", " ml ", "deep learning",
                            "training model", "tensorflow", "pytorch", "llm")),
    (UseCase.SOFTWARE_DEVELOPMENT, ("software development", "developer", "programming",
                                    "coding", "code", "docker", "kubernetes", "compile",
                                    "ide", "engineering student", "full stack")),
    (UseCase.CONTENT_CREATION, ("video editing", "photo editing", "content creation",
                                "premiere", "photoshop", "after effects", "rendering",
                                "3d", "cad", "blender", "streaming")),
    (UseCase.OFFICE_PRODUCTIVITY, ("office", "work", "business", "excel", "spreadsheet",
                                   "presentation", "meetings", "wfh", "productivity")),
    (UseCase.STUDENT, ("student", "college", "school", "study", "studies", "university")),
)

_KNOWN_BRANDS: frozenset[str] = frozenset(
    {"dell", "hp", "lenovo", "asus", "acer", "apple", "msi", "samsung", "lg",
     "microsoft", "infinix", "realme", "honor", "gigabyte", "razer"}
)

#: Compiled with word boundaries. Substring matching misfires badly here —
#: "video" contains "ide", "streaming" contains "ram".
_USE_CASE_PATTERNS: tuple[tuple[UseCase, re.Pattern[str]], ...] = tuple(
    (
        use_case,
        re.compile(
            "|".join(rf"\b{re.escape(term.strip())}\b" for term in terms),
            re.IGNORECASE,
        ),
    )
    for use_case, terms in _USE_CASE_TERMS
)

#: Phrases that make a nearby requirement non-negotiable.
_MANDATORY_MARKERS = re.compile(
    r"(?i)\b(?:must|mandatory|non[-\s]?negotiable|at\s+least|minimum|min\.?|"
    r"no\s+less\s+than|strictly|only|need\s+at\s+least|has\s+to|have\s+to|required)\b"
)

#: Scale words, always with the optional plural. "2 Lakhs budget" is the normal
#: way to say this, and a trailing \b after "lakh" cannot match the "s".
_SCALE_WORD = r"(?:k|lakhs?|lacs?|crores?)"
#: Capturing variant, for the pattern that reads the scale back out.
_SCALE_WORD_CAP = r"(k|lakhs?|lacs?|crores?)"

_BUDGET_CEILING = re.compile(
    r"(?i)\b(?:under|below|less\s+than|within|upto|up\s+to|max(?:imum)?|"
    r"budget\s+(?:of|is)?|around|about|near|approx\w*|not\s+more\s+than)\b[^.\d]{0,12}"
    r"([₹$]?\s*[\d,]+(?:\.\d+)?)\s*" + _SCALE_WORD_CAP + r"?"
)
#: A bare amount, including the very common "<amount> <scale> budget" ordering
#: where the keyword follows rather than precedes the number.
_BARE_AMOUNT = re.compile(
    r"(?i)([₹$]\s*[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*" + _SCALE_WORD
    + r"|\b[\d,]{4,}\b)"
)

_SCALE = {"k": Decimal("1000"), "lakh": Decimal("100000"), "lac": Decimal("100000"),
          "crore": Decimal("10000000")}

_RAM = re.compile(r"(?i)\b(\d{1,3})\s*gb\b(?:[^.\n]{0,12}\bram\b)?")
_RAM_EXPLICIT = re.compile(r"(?i)\b(\d{1,3})\s*gb\s*(?:of\s+)?ram\b|\bram\b[^.\d\n]{0,10}(\d{1,3})\s*gb")
_STORAGE_GB = re.compile(r"(?i)\b(\d{3,4})\s*gb\b[^.\n]{0,14}\b(?:ssd|hdd|storage|nvme)\b")
_STORAGE_TB = re.compile(r"(?i)\b(\d(?:\.\d)?)\s*tb\b")
_BARE_STORAGE = re.compile(r"(?i)\b(256|512|1024|2048)\s*gb\b")
_SCREEN = re.compile(r"(?i)\b(\d{2}(?:\.\d)?)\s*(?:inch|inches|\")")
_WEIGHT = re.compile(r"(?i)\b(\d(?:\.\d{1,2})?)\s*kg\b")
_BATTERY = re.compile(r"(?i)\b(\d{1,2})\s*(?:\+\s*)?(?:hour|hours|hrs?)\b[^.\n]{0,16}\bbattery\b"
                      r"|\bbattery\b[^.\n]{0,16}\b(\d{1,2})\s*(?:\+\s*)?(?:hour|hours|hrs?)\b")


#: Brands whose names are acronyms and look wrong title-cased.
_UPPERCASE_BRANDS = frozenset({"hp", "asus", "msi", "lg", "acer"})


def _both(*values: object) -> bool:
    """True only when every value is known."""
    return all(value is not None for value in values)


def _brand_display(brand: str) -> str:
    return brand.upper() if brand.lower() in _UPPERCASE_BRANDS else brand.title()


def _to_amount(raw: str, scale: str | None) -> Decimal | None:
    cleaned = re.sub(r"[₹$,\s]", "", raw)
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except Exception:
        return None
    if scale:
        # "Lakhs" -> "lakh": the plural is stripped before the lookup.
        value *= _SCALE[scale.lower().rstrip("s")]
    elif value < Decimal("1000"):
        # "budget 80" almost certainly means 80k in this domain.
        value *= Decimal("1000")
    return value if value > 0 else None


class OfflineReasoner:
    """Rule-based stand-in for the model, producing the same schemas."""

    # ------------------------------------------------------------------
    # requirements
    # ------------------------------------------------------------------

    def extract_requirements(
        self, text: str, *, previous: RequirementExtraction | None = None
    ) -> RequirementExtraction:
        lowered = f" {text.lower()} "
        data: dict[str, object] = {}
        mandatory: list[str] = []

        # ---- use case ----
        for use_case, pattern in _USE_CASE_PATTERNS:
            if pattern.search(lowered):
                data["use_case"] = use_case
                break

        # ---- budget ----
        budget = self._extract_budget(text)
        if budget is not None:
            data["budget_max"] = budget
            mandatory.append("budget_max")

        # ---- ram ----
        ram = self._first_int(_RAM_EXPLICIT, text)
        if ram is None:
            # A bare "16GB" not adjacent to a storage unit is RAM in practice.
            candidate = self._first_int(_RAM, text)
            if candidate is not None and candidate <= 128:
                ram = candidate
        if ram is not None and 2 <= ram <= 256:
            data["min_ram_gb"] = ram
            if _MANDATORY_MARKERS.search(text) or "ram" in lowered:
                mandatory.append("min_ram_gb")

        # ---- storage ----
        storage = self._first_int(_STORAGE_GB, text)
        if storage is None:
            tb = _STORAGE_TB.search(text)
            if tb:
                storage = int(Decimal(tb.group(1)) * 1024)
        if storage is None:
            # A bare common storage size with no "ssd"/"storage" word nearby.
            # Restricted to real capacities so "16GB" is never read as storage.
            bare = _BARE_STORAGE.search(text)
            if bare:
                storage = int(bare.group(1))
        if storage is not None and 32 <= storage <= 8192:
            data["min_storage_gb"] = storage
        if "ssd" in lowered:
            data["storage_type"] = "ssd"
        elif "hdd" in lowered:
            data["storage_type"] = "hdd"

        # ---- portability ----
        if any(term in lowered for term in (" light", "lightweight", "portable",
                                            "travel", "carry", "thin and light")):
            data["max_weight_kg"] = 1.6
            data.setdefault("max_screen_inches", 14.5)
        weight = _WEIGHT.search(text)
        if weight:
            value = float(weight.group(1))
            if 0.5 <= value <= 6.0:
                data["max_weight_kg"] = value
                if _MANDATORY_MARKERS.search(text):
                    mandatory.append("max_weight_kg")

        # ---- screen ----
        screen = _SCREEN.search(text)
        if screen:
            value = float(screen.group(1))
            if 8.0 <= value <= 20.0:
                if any(term in lowered for term in ("at least", "minimum", "bigger", "larger")):
                    data["min_screen_inches"] = value
                else:
                    data["min_screen_inches"] = max(8.0, value - 0.6)
                    data["max_screen_inches"] = min(20.0, value + 0.6)

        # ---- battery ----
        battery = self._first_int(_BATTERY, text)
        if battery is not None and 1 <= battery <= 30:
            data["min_battery_hours"] = float(battery)
        elif any(term in lowered for term in ("long battery", "battery life",
                                              "all day battery", "good battery")):
            data["min_battery_hours"] = 8.0

        # ---- os ----
        if "macos" in lowered or "macbook" in lowered or " mac " in lowered:
            data["required_os"] = "macos"
        elif "windows" in lowered:
            data["required_os"] = "windows"
        elif "linux" in lowered or "ubuntu" in lowered:
            data["required_os"] = "linux"
        elif "chromebook" in lowered or "chromeos" in lowered:
            data["required_os"] = "chromeos"

        # ---- gpu ----
        if any(term in lowered for term in ("dedicated gpu", "discrete gpu", "graphics card",
                                            "rtx", "gtx", "radeon rx", "nvidia")):
            data["dedicated_gpu_required"] = True
            mandatory.append("dedicated_gpu_required")
        elif data.get("use_case") in {UseCase.GAMING, UseCase.DATA_SCIENCE}:
            data["dedicated_gpu_required"] = True

        if "touchscreen" in lowered or "touch screen" in lowered:
            data["touchscreen_required"] = True

        # ---- brands ----
        preferred = [brand for brand in sorted(_KNOWN_BRANDS) if f" {brand} " in lowered]
        excluded = [
            brand
            for brand in sorted(_KNOWN_BRANDS)
            if re.search(rf"(?i)\b(?:not|no|avoid|except|other\s+than)\b[^.\n]{{0,20}}\b{brand}\b", lowered)
        ]
        preferred = [brand for brand in preferred if brand not in excluded]
        if preferred:
            data["preferred_brands"] = preferred
        if excluded:
            data["excluded_brands"] = excluded

        # ---- offer eligibility (bank name only, never digits) ----
        banks = [bank for bank in Bank if bank.value in lowered]
        if banks:
            data["eligible_banks"] = banks
        if "exchange" in lowered:
            data["has_exchange_device"] = True
        if "emi" in lowered:
            data["wants_no_cost_emi"] = True

        # ---- merge with what we already knew ----
        if previous is not None:
            merged = previous.model_dump()
            merged.update({key: value for key, value in data.items() if value is not None})
            merged["mandatory_fields"] = sorted(
                set(previous.mandatory_fields) | set(mandatory)
            )
            data = merged
        else:
            data["mandatory_fields"] = sorted(set(mandatory))

        data["confidence"] = self._confidence(data)
        extraction = RequirementExtraction.model_validate(data)
        # Drop mandatory markers for fields that ended up unset.
        return extraction.model_copy(
            update={
                "mandatory_fields": [
                    field
                    for field in extraction.mandatory_fields
                    if getattr(extraction, field, None) not in (None, [], False)
                ]
            }
        )

    def _extract_budget(self, text: str) -> ExtractedBudget | None:
        currency = "USD" if "$" in text or "usd" in text.lower() else "INR"
        match = _BUDGET_CEILING.search(text)
        if match:
            amount = _to_amount(match.group(1), match.group(2))
            if amount is not None:
                return ExtractedBudget(amount=float(amount), currency=currency)
        # A bare large number in a shopping request is a budget.
        bare = _BARE_AMOUNT.search(text)
        if bare:
            raw = bare.group(1)
            lowered = raw.lower()
            scale = next((unit for unit in ("lakh", "lac", "crore", "k") if unit in lowered), None)
            amount = _to_amount(re.sub(r"(?i)(k|lakhs?|lacs?|crores?)", "", raw), scale)
            if amount is not None and amount >= Decimal("10000"):
                return ExtractedBudget(amount=float(amount), currency=currency)
        return None

    @staticmethod
    def _first_int(pattern: re.Pattern[str], text: str) -> int | None:
        match = pattern.search(text)
        if not match:
            return None
        for group in match.groups():
            if group:
                try:
                    return int(Decimal(group))
                except Exception:
                    continue
        return None

    @staticmethod
    def _confidence(data: dict[str, object]) -> str:
        signals = sum(
            1
            for key in ("budget_max", "min_ram_gb", "min_storage_gb", "max_weight_kg",
                        "min_battery_hours", "dedicated_gpu_required")
            if data.get(key)
        )
        has_use_case = data.get("use_case") not in (None, UseCase.GENERAL)
        if signals >= 3 and has_use_case:
            return "high"
        if signals >= 1 or has_use_case:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # clarification
    # ------------------------------------------------------------------

    def decide_clarification(
        self, requirements: LaptopRequirements, missing: list[str]
    ) -> ClarificationDecision:
        """Ask only when the missing field would change the recommendation."""
        blocking = [field for field in missing if field in {"budget_max", "use_case"}]
        if not blocking:
            return ClarificationDecision(needs_clarification=False)

        if set(blocking) == {"budget_max"}:
            question = "What's your maximum budget for the laptop?"
        elif set(blocking) == {"use_case"}:
            question = "What will you mainly use the laptop for?"
        else:
            question = (
                "What's your maximum budget, and what will you mainly use the laptop for?"
            )
        return ClarificationDecision(
            needs_clarification=True, question=question, missing_fields=blocking
        )

    # ------------------------------------------------------------------
    # search terms
    # ------------------------------------------------------------------

    def decide_search(self, requirements: LaptopRequirements) -> SearchDecision:
        parts = ["laptop"]
        if requirements.min_ram_gb:
            parts.append(f"{requirements.min_ram_gb}GB")
        if requirements.min_storage_gb:
            parts.append(f"{requirements.min_storage_gb}GB SSD")
        if requirements.dedicated_gpu_required:
            parts.append("dedicated graphics")
        if requirements.use_case is not UseCase.GENERAL:
            parts.append(requirements.use_case.value.replace("_", " "))
        for brand in requirements.preferred_brands[:2]:
            parts.append(brand)
        query = " ".join(parts)[:120]
        return SearchDecision(
            query=query,
            alternate_queries=["laptop " + requirements.use_case.value.replace("_", " ")][:1],
            rationale="Derived deterministically from the validated requirements.",
        )

    # ------------------------------------------------------------------
    # explanation
    # ------------------------------------------------------------------

    def explain_recommendation(
        self,
        winner: ProductCandidate,
        runner_ups: list[ProductCandidate],
        requirements: LaptopRequirements,
    ) -> RecommendationExplanation:
        """Compose prose from the structured candidate data.

        Note that no figure is emitted — price is described qualitatively, exactly
        as the live prompt requires, so both modes go through the same
        price-claim screening unchanged.
        """
        specs = winner.product.specs
        name = winner.product.title.split(",")[0].strip()

        # Only specifications the marketplace actually reported are described.
        # A missing value is left unmentioned, never filled in.
        stated: list[str] = []
        if specs.ram_gb is not None:
            stated.append(f"{specs.ram_gb}GB RAM")
        if specs.storage_gb is not None:
            unit = f"{specs.storage_gb}GB"
            stated.append(
                f"{unit} {specs.storage_type.upper()}" if specs.storage_type else unit
            )
        if specs.screen_inches is not None:
            stated.append(f"a {specs.screen_inches}-inch display")
        if specs.cpu:
            stated.append(specs.cpu)

        if stated:
            sentences = [f"The {name} matches what you asked for: {', '.join(stated)}."]
        else:
            sentences = [
                f"The {name} ranked highest against your requirements on price and "
                "marketplace rating."
            ]

        if requirements.budget_max is not None:
            sentences.append("It stays within your budget after applicable discounts.")
        if requirements.dedicated_gpu_required and specs.dedicated_gpu:
            gpu = specs.gpu or "dedicated graphics"
            sentences.append(f"It includes the {gpu} you require.")
        if (
            requirements.min_battery_hours
            and specs.battery_hours is not None
            and specs.battery_hours >= requirements.min_battery_hours
        ):
            sentences.append(f"Rated battery life is around {specs.battery_hours} hours.")
        unknown = {"ram_gb", "storage_gb", "screen_inches", "weight_kg", "battery_hours"} - specs.known_fields()
        if unknown:
            sentences.append(
                "Some specifications were not listed by the marketplace, so they are "
                "shown as not stated rather than estimated."
            )
        if winner.price.unmet_conditional_offers:
            sentences.append(
                "Additional conditional offers are listed that you would need to qualify for."
            )
        if winner.price.cashback_value.amount > 0:
            sentences.append(
                "Cashback is offered separately and is returned after purchase rather "
                "than reducing the amount you pay now."
            )

        trade_offs: list[TradeOffOut] = []
        seen_dimensions: set[str] = set()

        def add(dimension: str, detail: str) -> None:
            # One trade-off per dimension: three lines all saying "lighter" is
            # noise, not information.
            if dimension not in seen_dimensions:
                seen_dimensions.add(dimension)
                trade_offs.append(TradeOffOut(dimension=dimension, detail=detail))

        for other in runner_ups[:3]:
            other_specs = other.product.specs
            brand = _brand_display(other.product.brand)
            # Each comparison needs both values known; an unknown is not a
            # difference worth asserting.
            if _both(other_specs.ram_gb, specs.ram_gb) and other_specs.ram_gb > specs.ram_gb:
                add(
                    "memory",
                    f"{brand} offers more RAM ({other_specs.ram_gb}GB) "
                    "but ranked lower overall.",
                )
            if (
                _both(other_specs.weight_kg, specs.weight_kg)
                and other_specs.weight_kg < specs.weight_kg - 0.3
            ):
                add(
                    "portability",
                    f"The {brand} option is lighter at {other_specs.weight_kg}kg.",
                )
            if other_specs.dedicated_gpu and not specs.dedicated_gpu:
                add("graphics", f"The {brand} option adds dedicated graphics.")
            if (
                _both(other_specs.battery_hours, specs.battery_hours)
                and other_specs.battery_hours > specs.battery_hours + 2
            ):
                add(
                    "battery life",
                    f"The {brand} option is rated for longer battery life "
                    f"({other_specs.battery_hours} hours).",
                )

        notes = [
            RunnerUpNote(
                product_id=other.product.product_id,
                why_not="Scored lower on the weighted fit, value and specification components.",
            )
            for other in runner_ups[:4]
        ]

        return RecommendationExplanation(
            rationale=" ".join(sentences)[:1200],
            trade_offs=trade_offs[:6],
            runner_up_notes=notes,
        )
