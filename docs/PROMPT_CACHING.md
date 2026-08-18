# Prompt caching and caching policy

Two unrelated things get called "caching" in this application. They are not
equivalent, and conflating them is the most common way to claim a cost saving
that does not exist.

## A. Provider-side prompt caching — the one that saves tokens

Anthropic caches the **tokenised prompt prefix** server-side. We opt in by
marking content blocks with `cache_control`. Cache reads cost roughly 0.1× the
normal input price; a cache write costs about 1.25× (5-minute TTL) or 2× (1-hour
TTL). This is a real reduction in cost and time-to-first-token.

It works only if the marked prefix is **byte-identical** between requests. The
cache key is a prefix match, so a single changed byte anywhere invalidates every
breakpoint after it.

## B. Local prompt-assembly caching — the one that saves microseconds

`CachedPromptProvider` memoises our own string concatenation. It avoids
rebuilding the same text, and — far more usefully — **guarantees the bytes are
identical every time**, which is what makes (A) actually hit.

It does **not** reduce tokens, cost or provider latency. Claiming otherwise would
be wrong. Its value is prefix stability, not throughput.

`assembly_hits` / `assembly_misses` measure (B). `cache_read_input_tokens` from
the model response measures (A). They are reported separately and never summed.

## Layout

```
system[0]  stable shared prefix          <- cache_control  (shared by ALL tasks)
           identity
           safety rules
           laptop domain rules
           scoring explanation
           output contract
system[1]  stable per-task instructions  <- cache_control  (shared by all calls of one task)
─────────────────────────────────────────── everything below is per-request, uncached
human      user requirements
           marketplace data
           conversation state
```

Two breakpoints, nested. The large shared prefix is written once and read by
every task; each task's instructions cache independently. Four breakpoints are
available per request; we use two.

The shared prefix is ~1,300 tokens, comfortably above Claude Opus 5's 512-token
minimum cacheable prefix. Below that minimum a marked prefix silently never
caches — no error, just `cache_creation_input_tokens: 0`. This is asserted in
`test_caching.py::test_prefix_exceeds_the_minimum_cacheable_size`.

## What is never cached

Prices and offers must always be fresh. The stable prefix contains, by
construction, none of:

- user requirements, budgets or card eligibility
- current prices, offers or discounts
- session identifiers or conversation state
- product results
- timestamps, dates or UUIDs

`test_prefix_contains_no_request_specific_content` asserts this rather than
trusting the convention.

## Keeping the prefix stable

The failure mode is silent: the cache stops hitting and nothing errors. Four
defences:

1. **The blocks are frozen constants.** `stable_prefix_blocks()` returns a fixed
   tuple in a fixed order. Callers must not filter or reorder it — doing so
   per-request would create one distinct prefix per variant.
2. **No conditional assembly.** There is no `if flag: system += ...`. Every flag
   combination would be a separate prefix.
3. **Fresh dict objects, memoised text.** `system_blocks()` returns new dicts each
   call so a mutating consumer cannot corrupt the shared text, while the text
   itself is the memoised identical string.
4. **A fingerprint.** `prefix_fingerprint()` is a digest of the prefix, exposed on
   `GET /health` and attached to every trace. If it changes between deployments,
   every cache entry was invalidated — visible immediately rather than as a
   mysteriously higher bill.

Other invalidators to avoid: changing the model (caches are model-scoped),
changing the tool set (tools render before system), and non-deterministic JSON
serialisation. `wrap_untrusted` sorts keys for that last reason.

## Verifying it works

`cache_read_input_tokens` is the only number that proves (A) is working. If it
stays zero across repeated requests with the same prefix, something is
invalidating it. It is surfaced in three places: `InvocationStats`,
`RunMetrics.as_trace_metadata()`, and the Streamlit diagnostics panel.

Note that `input_tokens` reports only the *uncached remainder*. Total prompt size
is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.

## Prompt versioning

Prompts live in `src/laptop_agent/prompts/`, separate from application code:

```
prompts/
  base.py            BASE_PROMPT_VERSION            stable prefix
  requirements.py    REQUIREMENTS_PROMPT_VERSION    extraction + clarification
  recommendation.py  RECOMMENDATION_PROMPT_VERSION  explanation
  provider.py        assembly, breakpoints, versioning
```

Every version is attached to LangSmith metadata, so comparing prompt v1 against
v2 is unambiguous:

```json
{
  "base_prompt_version": "v1",
  "task_prompt_version": "v1",
  "prompt_task": "requirements",
  "prompt_prefix_fingerprint": "54b73cbe78dff4f9",
  "prompt_caching_enabled": "true",
  "prompt_cache_ttl": "5m"
}
```

Bump the version when the text changes. The fingerprint will change too, which is
the signal that cache entries were rotated deliberately rather than by accident.

## PromptProvider abstraction

```python
class PromptProvider(Protocol):
    def get_requirements_prompt(self) -> str: ...
    def get_recommendation_prompt(self) -> str: ...
    def system_blocks(self, task: PromptTask) -> list[dict]: ...
    def versions(self, task: PromptTask) -> dict[str, str]: ...
```

`CachedPromptProvider` is the implementation. `system_blocks()` is the
provider-aware path (block-level `cache_control`); `system_text()` is the
flattened fallback for a provider without block-level caching, which gets **no
token caching at all** — only assembly stability.

## C. Product and search caching

An entirely separate concern, and the rules are the opposite way round: this data
is **volatile**.

| Data | TTL | Rationale |
|------|-----|-----------|
| Product search results | 300 s (5 min) | Listings change; prices change faster |
| Offers | 120 s (1–5 min) | Offers rotate frequently |
| Prices | never cached beyond the above | Treated as volatile by policy |

Enforced in two places, not just documented:

- `Settings` **rejects** a configured TTL above 300 seconds — a validation error
  at startup, not a warning.
- `InMemoryCacheProvider` **clamps** any TTL passed to `set()` to the same
  ceiling, so no caller can pin a stale price.

```python
class CacheProvider(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_seconds: int) -> None: ...
    def delete(self, key: str) -> None: ...
```

`InMemoryCacheProvider` is the only implementation: thread-safe, TTL-bounded,
size-capped. **Redis is deliberately absent** — it would add an operational
dependency for data whose maximum useful lifetime is five minutes. When something
genuinely needs it, implement `CacheProvider` and inject it; no calling code
changes.

Cache keys include every argument that affects the response (query, marketplace,
category, max_results, currency, budget). Omitting one would serve a response for
a different query — `test_different_queries_do_not_share_a_cache_entry` guards it.
