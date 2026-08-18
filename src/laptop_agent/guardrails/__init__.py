"""Guardrails.

Layered, and deliberately independent of each other — each layer assumes the
ones before it may have been bypassed:

============================  =====================================================
Layer                         Module
============================  =====================================================
user input                    :mod:`.input_guardrail`
domain scope                  :mod:`.scope_guardrail`
injection detection           :mod:`.injection`, :mod:`.patterns`
trust framing for the model   :mod:`.untrusted`
tool arguments                :mod:`.tool_input`
marketplace responses         :mod:`.tool_output`
prices                        :mod:`.price_validator`
LLM price claims in prose     :mod:`.price_claims`
final recommendation          :mod:`.recommendation_validator`
============================  =====================================================
"""

from .injection import InjectionScan, scan_for_injection, strip_control_characters
from .input_guardrail import InputGuardrail
from .price_claims import PriceClaimReport, screen_price_claims
from .price_validator import (
    SUPPORTED_CURRENCIES,
    PriceValidationReport,
    PriceValidator,
)
from .recommendation_validator import (
    ProviderRegistry,
    RecommendationValidationResult,
    RecommendationValidator,
    ValidationFailure,
)
from .result import (
    SAFE_RESPONSES,
    GuardrailAction,
    GuardrailResult,
    ValidationOutcome,
)
from .scope_guardrail import ConversationStage, ScopeGuardrail, ScopeVerdict
from .tool_input import (
    FetchOffersRequest,
    SearchProductsRequest,
    ToolArgumentError,
)
from .tool_output import (
    TRUSTED_HOSTS,
    MarketplaceResponseValidator,
    OfferPayload,
    ProductPayload,
)
from .untrusted import TrustLabel, wrap_untrusted

__all__ = [
    "ConversationStage",
    "FetchOffersRequest",
    "GuardrailAction",
    "GuardrailResult",
    "InjectionScan",
    "InputGuardrail",
    "MarketplaceResponseValidator",
    "OfferPayload",
    "PriceClaimReport",
    "PriceValidationReport",
    "PriceValidator",
    "ProductPayload",
    "ProviderRegistry",
    "RecommendationValidationResult",
    "RecommendationValidator",
    "SAFE_RESPONSES",
    "SUPPORTED_CURRENCIES",
    "ScopeGuardrail",
    "ScopeVerdict",
    "SearchProductsRequest",
    "TRUSTED_HOSTS",
    "ToolArgumentError",
    "TrustLabel",
    "ValidationFailure",
    "ValidationOutcome",
    "scan_for_injection",
    "screen_price_claims",
    "strip_control_characters",
    "wrap_untrusted",
]
