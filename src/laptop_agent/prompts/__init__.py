"""Versioned prompts, stored separately from application code.

Stable content lives in :mod:`.base` and the per-task modules; nothing here
interpolates request data. See :mod:`.provider` for assembly and cache
breakpoint placement.
"""

from .base import BASE_PROMPT_VERSION, stable_prefix_blocks
from .provider import (
    CachedPromptProvider,
    PromptProvider,
    PromptTask,
    get_prompt_provider,
    reset_prompt_provider,
)
from .recommendation import RECOMMENDATION_PROMPT_VERSION
from .requirements import CLARIFICATION_PROMPT_VERSION, REQUIREMENTS_PROMPT_VERSION

__all__ = [
    "BASE_PROMPT_VERSION",
    "CLARIFICATION_PROMPT_VERSION",
    "CachedPromptProvider",
    "PromptProvider",
    "PromptTask",
    "RECOMMENDATION_PROMPT_VERSION",
    "REQUIREMENTS_PROMPT_VERSION",
    "get_prompt_provider",
    "reset_prompt_provider",
    "stable_prefix_blocks",
]
