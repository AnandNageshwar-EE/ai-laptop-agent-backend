"""Tool argument validation.

Every tool argument is a Pydantic model with closed bounds. The important
property is what these models make *impossible*: there is no field anywhere that
accepts a URL, a host, a path or a request body. A marketplace client owns its
own base URL, so no model output can direct a request at an arbitrary endpoint.

``extra="forbid"`` everywhere means an unexpected argument is a validation
error, not a silently ignored field.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain.enums import Currency, Marketplace, ProductCategory
from ..domain.money import Money

#: Search queries are short keyword strings, not prose and not payloads.
MAX_QUERY_CHARS = 120
#: Absolute ceiling regardless of configuration.
MAX_RESULTS_HARD_LIMIT = 50


class SearchProductsRequest(BaseModel):
    """Arguments for a marketplace product search."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: Annotated[str, Field(min_length=2, max_length=MAX_QUERY_CHARS)]
    marketplace: Marketplace
    max_results: Annotated[int, Field(ge=1, le=MAX_RESULTS_HARD_LIMIT)] = 10
    category: ProductCategory = ProductCategory.LAPTOP
    budget_max: Money | None = None
    currency: Currency = Currency.INR

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        """Reduce the query to search-safe characters.

        This is not cosmetic. A query is forwarded to a provider, so anything
        that could be interpreted as syntax by a downstream system is removed
        rather than escaped — the character set is restricted to what a laptop
        search legitimately needs.
        """
        cleaned = "".join(
            char for char in value if char.isalnum() or char in " .,+-/&'\"()"
        )
        cleaned = " ".join(cleaned.split())
        if len(cleaned) < 2:
            raise ValueError("query has no searchable content after sanitisation")
        return cleaned

    @model_validator(mode="after")
    def _currency_agrees_with_budget(self) -> SearchProductsRequest:
        if self.budget_max is not None and self.budget_max.currency is not self.currency:
            raise ValueError("budget_max currency must match the request currency")
        return self


class FetchOffersRequest(BaseModel):
    """Arguments for fetching offers for already-retrieved products."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    marketplace: Marketplace
    #: Ids must come from products this run already retrieved — the caller passes
    #: them from validated provider results, never from model output.
    product_ids: Annotated[
        list[Annotated[str, Field(pattern=r"^[A-Za-z0-9._-]{4,64}$")]],
        Field(min_length=1, max_length=MAX_RESULTS_HARD_LIMIT),
    ]

    @field_validator("product_ids")
    @classmethod
    def _dedupe(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return seen


class ToolArgumentError(ValueError):
    """Raised when tool arguments fail validation.

    Carries the field-level detail for the audit log while the caller returns a
    generic message to the user.
    """

    def __init__(self, tool: str, errors: list[dict[str, object]]) -> None:
        self.tool = tool
        self.errors = errors
        super().__init__(f"invalid arguments for tool {tool!r}")
