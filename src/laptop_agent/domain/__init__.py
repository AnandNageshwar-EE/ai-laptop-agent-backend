"""Validated domain models. Nothing enters graph state except through these."""

from .enums import (
    Bank,
    Currency,
    Marketplace,
    OfferKind,
    ProductCategory,
    RejectionReason,
    UseCase,
)
from .money import CurrencyMismatchError, Money, MoneyInput
from .product import (
    AppliedOffer,
    LaptopSpecs,
    Offer,
    PriceBreakdown,
    Product,
    ProductCandidate,
    evaluate_hard_constraints,
)
from .recommendation import Recommendation, RunnerUp, TradeOff
from .requirements import (
    CONSTRAINABLE_FIELDS,
    LaptopRequirements,
    PurchaseProfile,
)
from .scoring import SCORE_EPSILON, SCORING_VERSION, ProductScore, ScoreComponent

__all__ = [
    "AppliedOffer",
    "Bank",
    "CONSTRAINABLE_FIELDS",
    "Currency",
    "CurrencyMismatchError",
    "LaptopRequirements",
    "LaptopSpecs",
    "Marketplace",
    "Money",
    "MoneyInput",
    "Offer",
    "OfferKind",
    "PriceBreakdown",
    "Product",
    "ProductCandidate",
    "ProductCategory",
    "ProductScore",
    "PurchaseProfile",
    "Recommendation",
    "RejectionReason",
    "RunnerUp",
    "SCORE_EPSILON",
    "SCORING_VERSION",
    "ScoreComponent",
    "TradeOff",
    "UseCase",
    "evaluate_hard_constraints",
]
