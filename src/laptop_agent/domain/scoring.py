"""Score models.

Scores must be *reproducible*: the recommendation validator recomputes the
score for the selected candidate and compares it against the stored value. That
only works if a score is a pure function of (candidate, requirements, weights,
version) — so the version and the component breakdown are part of the model.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import Marketplace

#: Bump when weights or component maths change, so old scores are never
#: silently compared against new ones.
SCORING_VERSION = "v1"

#: Tolerance when comparing a stored score against a recomputed one.
SCORE_EPSILON = Decimal("0.0001")


class ScoreComponent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=48)]
    weight: Annotated[Decimal, Field(ge=0, le=1)]
    #: Normalised 0..1 measurement before weighting.
    raw: Annotated[Decimal, Field(ge=0, le=1)]

    @property
    def weighted(self) -> Decimal:
        return (self.weight * self.raw).quantize(Decimal("0.000001"))


class ProductScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: str
    marketplace: Marketplace
    components: Annotated[list[ScoreComponent], Field(min_length=1)]
    total: Annotated[Decimal, Field(ge=0, le=1)]
    scoring_version: str = SCORING_VERSION

    @model_validator(mode="after")
    def _total_is_sum_of_components(self) -> ProductScore:
        expected = sum((c.weighted for c in self.components), start=Decimal("0"))
        expected = expected.quantize(Decimal("0.000001"))
        if abs(self.total - expected) > SCORE_EPSILON:
            raise ValueError(
                f"total {self.total} does not match component sum {expected}"
            )
        weight_sum = sum((c.weight for c in self.components), start=Decimal("0"))
        if abs(weight_sum - Decimal("1")) > Decimal("0.0001"):
            raise ValueError(f"component weights must sum to 1, got {weight_sum}")
        return self

    @property
    def key(self) -> tuple[Marketplace, str]:
        return (self.marketplace, self.product_id)

    def matches(self, other: ProductScore) -> bool:
        """Whether two scores agree — used for the reproducibility check."""
        return (
            self.key == other.key
            and self.scoring_version == other.scoring_version
            and abs(self.total - other.total) <= SCORE_EPSILON
        )
