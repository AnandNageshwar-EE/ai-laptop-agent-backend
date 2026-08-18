"""LLM access: schemas, provider construction, validated invocation."""

from .offline import OfflineReasoner
from .provider import LLMUnavailableError, build_chat_model, describe_model
from .schemas import (
    ClarificationDecision,
    ExtractedBudget,
    RecommendationExplanation,
    RequirementExtraction,
    RunnerUpNote,
    ScopeAssessment,
    SearchDecision,
    TradeOffOut,
)
from .structured import InvocationStats, StructuredLLM, StructuredOutputError

__all__ = [
    "ClarificationDecision",
    "ExtractedBudget",
    "InvocationStats",
    "LLMUnavailableError",
    "OfflineReasoner",
    "RecommendationExplanation",
    "RequirementExtraction",
    "RunnerUpNote",
    "ScopeAssessment",
    "SearchDecision",
    "StructuredLLM",
    "StructuredOutputError",
    "TradeOffOut",
]
