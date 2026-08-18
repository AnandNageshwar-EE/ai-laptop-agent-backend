"""User requirements and purchase eligibility profile.

Two rules are enforced structurally here:

1. Requirements marked mandatory are *hard* constraints. A product that fails
   one is never recommended, regardless of score.
2. The purchase profile stores eligibility booleans only. It has no field that
   can hold a card number, and a validator rejects long digit runs so a leaked
   number cannot be smuggled into a free-text field.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import Bank, Currency, UseCase
from .money import Money

StorageType = Literal["ssd", "hdd", "any"]
OperatingSystem = Literal["windows", "macos", "linux", "chromeos", "any"]

# Fields on LaptopRequirements that may be declared mandatory.
CONSTRAINABLE_FIELDS: frozenset[str] = frozenset(
    {
        "budget_max",
        "min_ram_gb",
        "min_storage_gb",
        "storage_type",
        "min_screen_inches",
        "max_screen_inches",
        "max_weight_kg",
        "min_battery_hours",
        "required_os",
        "dedicated_gpu_required",
        "touchscreen_required",
        "preferred_brands",
        "excluded_brands",
    }
)

_DIGIT_RUN = re.compile(r"\d[\d\s-]{10,}")


class LaptopRequirements(BaseModel):
    """Validated laptop requirements. Produced only via a validated LLM/rule
    extraction, never by assigning raw model text into state."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_case: UseCase = UseCase.GENERAL

    budget_min: Money | None = None
    budget_max: Money | None = None

    min_ram_gb: Annotated[int, Field(ge=2, le=256)] | None = None
    min_storage_gb: Annotated[int, Field(ge=32, le=8192)] | None = None
    storage_type: StorageType = "any"

    min_screen_inches: Annotated[float, Field(ge=8.0, le=20.0)] | None = None
    max_screen_inches: Annotated[float, Field(ge=8.0, le=20.0)] | None = None
    max_weight_kg: Annotated[float, Field(gt=0.2, le=6.0)] | None = None
    min_battery_hours: Annotated[float, Field(ge=1.0, le=30.0)] | None = None

    required_os: OperatingSystem = "any"
    dedicated_gpu_required: bool = False
    touchscreen_required: bool = False

    preferred_brands: Annotated[list[str], Field(max_length=8)] = Field(
        default_factory=list
    )
    excluded_brands: Annotated[list[str], Field(max_length=8)] = Field(
        default_factory=list
    )

    #: Subset of :data:`CONSTRAINABLE_FIELDS` that must be satisfied exactly.
    mandatory_fields: Annotated[list[str], Field(max_length=14)] = Field(
        default_factory=list
    )

    @field_validator("preferred_brands", "excluded_brands")
    @classmethod
    def _normalise_brands(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            token = re.sub(r"[^a-zA-Z0-9 .+-]", "", value).strip().lower()
            if token and len(token) <= 32 and token not in cleaned:
                cleaned.append(token)
        return cleaned

    @field_validator("mandatory_fields")
    @classmethod
    def _known_constraints(cls, values: list[str]) -> list[str]:
        unknown = sorted(set(values) - CONSTRAINABLE_FIELDS)
        if unknown:
            raise ValueError(f"unknown constrainable field(s): {unknown}")
        # Deterministic order keeps scoring and trace payloads reproducible.
        return sorted(set(values))

    @model_validator(mode="after")
    def _check_coherence(self) -> LaptopRequirements:
        if self.budget_min and self.budget_max:
            if self.budget_min.currency is not self.budget_max.currency:
                raise ValueError("budget_min and budget_max must share a currency")
            if self.budget_min > self.budget_max:
                raise ValueError("budget_min cannot exceed budget_max")
        if (
            self.min_screen_inches
            and self.max_screen_inches
            and self.min_screen_inches > self.max_screen_inches
        ):
            raise ValueError("min_screen_inches cannot exceed max_screen_inches")
        overlap = set(self.preferred_brands) & set(self.excluded_brands)
        if overlap:
            raise ValueError(f"brand cannot be both preferred and excluded: {sorted(overlap)}")
        # A mandatory constraint on an unset field is meaningless.
        for field in self.mandatory_fields:
            value = getattr(self, field)
            if value is None or (isinstance(value, list) and not value):
                raise ValueError(f"field {field!r} is mandatory but has no value")
        return self

    @property
    def currency(self) -> Currency:
        for budget in (self.budget_max, self.budget_min):
            if budget is not None:
                return budget.currency
        return Currency.INR

    @property
    def has_concrete_constraints(self) -> bool:
        """Whether any specific specification was asked for.

        Used to decide whether the use case still needs asking. "16GB RAM and an
        SSD under 80000" is perfectly searchable even though no use case was
        named — demanding one would be interrogating the user for information
        their request already made unnecessary.
        """
        return any(
            (
                self.min_ram_gb is not None,
                self.min_storage_gb is not None,
                self.min_screen_inches is not None,
                self.max_screen_inches is not None,
                self.max_weight_kg is not None,
                self.min_battery_hours is not None,
                self.dedicated_gpu_required,
                self.touchscreen_required,
                self.storage_type != "any",
                self.required_os != "any",
                bool(self.preferred_brands),
            )
        )

    @property
    def is_actionable(self) -> bool:
        """Enough signal to run a marketplace search without guessing."""
        return (
            self.budget_max is not None
            or self.use_case is not UseCase.GENERAL
            or self.has_concrete_constraints
        )

    def missing_for_search(self) -> list[str]:
        """Fields whose absence would materially change the recommendation.

        A budget is always worth asking for — it is the ceiling every candidate
        is filtered against. The use case is only worth asking for when the
        request carries no concrete constraints either, because otherwise there
        is already enough to rank on.
        """
        missing: list[str] = []
        if self.budget_max is None:
            missing.append("budget_max")
        if self.use_case is UseCase.GENERAL and not self.has_concrete_constraints:
            missing.append("use_case")
        return missing


class PurchaseProfile(BaseModel):
    """Offer eligibility. Deliberately holds no payment instrument data.

    ``"My HDFC card ends in 1234"`` becomes ``eligible_banks=[Bank.HDFC]``.
    The digits are dropped at the boundary and never reach this model.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    eligible_banks: Annotated[list[Bank], Field(max_length=len(Bank))] = Field(
        default_factory=list
    )
    has_exchange_device: bool = False
    wants_no_cost_emi: bool = False

    @field_validator("eligible_banks")
    @classmethod
    def _dedupe(cls, values: list[Bank]) -> list[Bank]:
        return sorted(set(values), key=lambda bank: bank.value)

    @model_validator(mode="after")
    def _reject_embedded_numbers(self) -> PurchaseProfile:
        """Defence in depth: no field here may carry a digit run.

        Every field is currently an enum or a bool, so this cannot trigger
        today. It exists so that adding a free-text field later fails loudly
        rather than quietly becoming a place card numbers can land.
        """
        for name, value in self:
            if isinstance(value, str) and _DIGIT_RUN.search(value):
                raise ValueError(
                    f"field {name!r} looks like it contains a payment instrument number"
                )
        return self

    def is_eligible_for(self, bank: Bank | None) -> bool:
        return bank is None or bank in self.eligible_banks

    @property
    def has_hdfc_card(self) -> bool:
        return Bank.HDFC in self.eligible_banks
