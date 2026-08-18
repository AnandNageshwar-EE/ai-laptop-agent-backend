"""Prompt assembly, versioning and caching.

Two genuinely different things are called "prompt caching". They are not
equivalent and this module keeps them distinct:

**1. Provider-side prompt caching (the one that saves tokens and latency).**
Anthropic caches the *tokenised* prompt prefix server-side. We opt in by marking
content blocks with ``cache_control``. Cache reads cost ~0.1x input price. This
is real cost reduction, and it only works if the marked prefix is byte-identical
between requests.

**2. Local prompt-assembly caching (the one that saves microseconds).**
Memoising our own string concatenation. It avoids rebuilding the same prompt
text repeatedly and — more usefully — *guarantees* the bytes are identical each
time, which is what makes (1) hit. It does not reduce tokens, cost or latency at
the provider. Claiming otherwise would be wrong.

Layout produced here, with two cache breakpoints:

    [ block 1: shared stable prefix        ] <- cache_control  (shared by ALL tasks)
    [ block 2: stable per-task instructions] <- cache_control  (shared by all calls of one task)
    ------------------------------------------ everything below is per-request, uncached
    [ human message: requirements, marketplace data, conversation state ]

Nesting the breakpoints this way means the large shared prefix is written to
cache once and read by every task, while each task's instructions are cached
independently. See ``docs/PROMPT_CACHING.md``.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .base import BASE_PROMPT_VERSION, stable_prefix_blocks
from .recommendation import RECOMMENDATION_PROMPT_VERSION, RECOMMENDATION_TASK_BLOCK
from .requirements import (
    CLARIFICATION_PROMPT_VERSION,
    CLARIFICATION_TASK_BLOCK,
    REQUIREMENTS_PROMPT_VERSION,
    REQUIREMENTS_TASK_BLOCK,
)

#: Separator between stable blocks. Constant — part of the cached bytes.
_JOIN = "\n\n"


class PromptTask(StrEnum):
    REQUIREMENTS = "requirements"
    CLARIFICATION = "clarification"
    RECOMMENDATION = "recommendation"


_TASK_BLOCKS: dict[PromptTask, str] = {
    PromptTask.REQUIREMENTS: REQUIREMENTS_TASK_BLOCK,
    PromptTask.CLARIFICATION: CLARIFICATION_TASK_BLOCK,
    PromptTask.RECOMMENDATION: RECOMMENDATION_TASK_BLOCK,
}

_TASK_VERSIONS: dict[PromptTask, str] = {
    PromptTask.REQUIREMENTS: REQUIREMENTS_PROMPT_VERSION,
    PromptTask.CLARIFICATION: CLARIFICATION_PROMPT_VERSION,
    PromptTask.RECOMMENDATION: RECOMMENDATION_PROMPT_VERSION,
}


@runtime_checkable
class PromptProvider(Protocol):
    """Source of stable prompt text.

    Prompts live in this package, separate from application code, and each is
    versioned so ``prompt v1`` vs ``prompt v2`` can be compared unambiguously in
    LangSmith.
    """

    def get_requirements_prompt(self) -> str: ...

    def get_clarification_prompt(self) -> str: ...

    def get_recommendation_prompt(self) -> str: ...

    def system_blocks(self, task: PromptTask) -> list[dict[str, Any]]: ...

    def versions(self, task: PromptTask) -> dict[str, str]: ...


class CachedPromptProvider:
    """Assembles prompts once and reuses the exact same strings thereafter.

    The memo is what guarantees byte-stability of the cached prefix: every call
    returns the identical object, so no request can accidentally differ by a
    whitespace change or a reordered block.
    """

    def __init__(self, *, caching_enabled: bool = True, cache_ttl: str = "5m") -> None:
        self._caching_enabled = caching_enabled
        if cache_ttl not in {"5m", "1h"}:
            raise ValueError(f"unsupported prompt cache TTL: {cache_ttl!r}")
        self._cache_ttl = cache_ttl
        self._assembly_cache: dict[str, Any] = {}
        #: Local assembly-cache counters. Distinct from provider cache hits,
        #: which are read from the model response's usage metadata.
        self.assembly_hits = 0
        self.assembly_misses = 0

    # ----- stable prefix -----

    def stable_prefix(self) -> str:
        """The shared prefix used by every task. Byte-stable for the process life."""
        cached = self._assembly_cache.get("__prefix__")
        if cached is not None:
            self.assembly_hits += 1
            return cached
        self.assembly_misses += 1
        prefix = _JOIN.join(stable_prefix_blocks())
        self._assembly_cache["__prefix__"] = prefix
        return prefix

    def prefix_fingerprint(self) -> str:
        """Digest of the stable prefix.

        Recorded in trace metadata. If this value changes between deployments,
        every provider-side cache entry was invalidated — which is the single
        most common cause of a silently collapsed cache hit rate.
        """
        return hashlib.sha256(self.stable_prefix().encode("utf-8")).hexdigest()[:16]

    # ----- per-task prompts -----

    def _task_prompt(self, task: PromptTask) -> str:
        key = f"task::{task.value}"
        cached = self._assembly_cache.get(key)
        if cached is not None:
            self.assembly_hits += 1
            return cached
        self.assembly_misses += 1
        prompt = _TASK_BLOCKS[task]
        self._assembly_cache[key] = prompt
        return prompt

    def get_requirements_prompt(self) -> str:
        return self._task_prompt(PromptTask.REQUIREMENTS)

    def get_clarification_prompt(self) -> str:
        return self._task_prompt(PromptTask.CLARIFICATION)

    def get_recommendation_prompt(self) -> str:
        return self._task_prompt(PromptTask.RECOMMENDATION)

    # ----- provider-facing assembly -----

    def system_blocks(self, task: PromptTask) -> list[dict[str, Any]]:
        """System content blocks with provider cache breakpoints.

        Returns a fresh list of fresh dicts each call — LangChain and the SDK may
        mutate the structures they are handed, and a shared mutable block would
        let one request corrupt the "stable" text for the next. The *text* is the
        memoised, byte-identical string, which is what the cache key depends on.
        """
        cache_control = (
            {"type": "ephemeral", **({"ttl": "1h"} if self._cache_ttl == "1h" else {})}
            if self._caching_enabled
            else None
        )

        prefix_block: dict[str, Any] = {"type": "text", "text": self.stable_prefix()}
        task_block: dict[str, Any] = {"type": "text", "text": self._task_prompt(task)}

        if cache_control is not None:
            # Breakpoint 1: the prefix shared by every task.
            prefix_block["cache_control"] = dict(cache_control)
            # Breakpoint 2: prefix + this task's instructions.
            task_block["cache_control"] = dict(cache_control)

        return [prefix_block, task_block]

    def system_text(self, task: PromptTask) -> str:
        """Flattened system prompt, for providers without block-level caching."""
        return _JOIN.join(block["text"] for block in self.system_blocks(task))

    # ----- versioning / observability -----

    def versions(self, task: PromptTask) -> dict[str, str]:
        """Prompt versions for LangSmith metadata (spec section 5.1)."""
        return {
            "base_prompt_version": BASE_PROMPT_VERSION,
            "task_prompt_version": _TASK_VERSIONS[task],
            "prompt_task": task.value,
            "prompt_prefix_fingerprint": self.prefix_fingerprint(),
            "prompt_caching_enabled": str(self._caching_enabled).lower(),
            "prompt_cache_ttl": self._cache_ttl,
        }

    def assembly_stats(self) -> dict[str, int | float]:
        total = self.assembly_hits + self.assembly_misses
        return {
            "assembly_hits": self.assembly_hits,
            "assembly_misses": self.assembly_misses,
            "assembly_hit_rate": round(self.assembly_hits / total, 4) if total else 0.0,
        }


_default_provider: CachedPromptProvider | None = None


def get_prompt_provider() -> CachedPromptProvider:
    """Process-wide provider.

    Deliberately a singleton: a fresh provider per request would still produce
    identical bytes, but sharing one instance makes that guarantee structural
    and keeps the assembly cache warm.
    """
    global _default_provider
    if _default_provider is None:
        from ..config import get_settings

        settings = get_settings()
        _default_provider = CachedPromptProvider(
            caching_enabled=settings.prompt_caching_enabled,
            cache_ttl=settings.prompt_cache_ttl,
        )
    return _default_provider


def reset_prompt_provider() -> None:
    """Test helper."""
    global _default_provider
    _default_provider = None
