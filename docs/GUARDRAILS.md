# Guardrails

Everything outside this application's own source is untrusted: user input, LLM
output, marketplace content, tool arguments and external API responses. The
prompt is not a security control — it is one layer among several, and every
other layer assumes it may have failed.

## Layers

| # | Layer | Module | Failure mode it prevents |
|---|-------|--------|--------------------------|
| 1 | Input validation | `guardrails/input_guardrail.py` | Oversized, empty, malformed or hostile input entering the graph |
| 2 | Domain scope | `guardrails/scope_guardrail.py` | The agent being repurposed for malware, intrusion, fraud or unrelated advice |
| 3 | Injection detection | `guardrails/injection.py`, `guardrails/patterns.py` | Instruction override, prompt extraction, role spoofing, jailbreaks, secret exfiltration, recommendation manipulation |
| 4 | Trust framing | `guardrails/untrusted.py` | Untrusted text being read as instruction by the model |
| 5 | Tool arguments | `guardrails/tool_input.py` | The model directing requests at arbitrary endpoints, or unbounded result counts |
| 6 | Tool responses | `guardrails/tool_output.py` | Malformed products, bad prices, off-platform URLs becoming state |
| 7 | Prices | `guardrails/price_validator.py` | Double discounts, assumed conditional offers, cashback treated as upfront |
| 8 | LLM price claims | `guardrails/price_claims.py` | Invented monetary figures in model prose |
| 9 | Final recommendation | `guardrails/recommendation_validator.py` | Any of the above having failed silently |

The layers are independent. Layer 9 re-derives the price, the constraints and the
score from provider data rather than trusting layers 6–8 to have worked.

## 1. Input guardrail

Checks run cheapest-first, so a large payload is rejected on length before any
regex work. In order: type, size, character hygiene, emptiness, content shape,
injection scan, secret stripping.

Returns a **sanitised** string. Downstream code uses that value, so control
characters and invisible formatting never reach the model even on an allowed
request.

Control characters are treated as a sanitisation signal rather than an attack —
they are usually a copy-paste artefact. Injection *wording* hidden by them is
still caught, because the scanner screens a control-stripped variant of the text
(`patterns.matching_variants`).

## 2. Scope guardrail

Two separate decisions with different failure costs:

- **Disallowed topic** (malware, intrusion, fraud, harm, unrelated professional
  advice) — deterministic deny, evaluated on every turn regardless of
  conversation state. A false negative here is a security incident.
- **Off-topic** — evaluated against the conversation stage. A terse answer to a
  clarifying question (`"yes"`, `"80k"`, `"hdfc"`) shares no vocabulary with the
  laptop lexicon, so a naive relevance check would reject the user's own reply.
  `ConversationStage` carries that context.

`"stock"` is deliberately ambiguous and handled explicitly: `"is it in stock"` is
shopping vocabulary, `"stocks to buy"` is financial advice. Only
finance-specific constructions are refused.

An LLM classifier may act as a tie-breaker on the genuinely unclear middle, but
it is **advisory only** — it can never overturn a deterministic deny, and if it
is unavailable the deterministic verdict stands.

## 3. Prompt injection

Detection is regex and set membership, never a model call — an LLM classifier is
itself a target of the injection it is meant to detect.

Six categories are tracked (`patterns.INJECTION_CATEGORIES`), and each maps to a
rejection reason. Audit records name the *category*, never the matched span,
because logging the span would put attacker-controlled text into the trace.

Obfuscation is handled by screening several normalised variants of the same text:
NFKC-normalised, invisible characters deleted, invisible characters replaced by a
space, and control characters stripped. That closes both directions of
zero-width evasion (`ignore<ZWSP>all<ZWSP>previous` and `ig<ZWSP>nore`).

### Marketplace content is treated differently from user input

The same scanner runs over seller-authored text, but the response differs:

- User input that attacks the system is **blocked** — the turn stops.
- Marketplace text that attacks the system is **kept as data, flagged, and the
  listing is disqualified from being recommended** — the search continues.

Three distinct decisions, and it is worth separating them because an earlier
version of this design conflated the second and third:

1. **Does hostile text break the search?** No. The listing is validated and
   processed like any other, so one poisoned description cannot deny service to
   the user or make a whole marketplace response fail.
2. **Is the text ever followed?** No. It is wrapped as untrusted data, and it is
   also neutralised before display (`guardrails/display.py`) so the payload is
   not echoed onto the user's screen.
3. **Can the listing still win the recommendation?** **No.** This is the part
   that was originally wrong. "Don't let hostile text break the search" does not
   imply "the listing stays eligible to win". A seller controls their own
   description, so demoting it penalises the party responsible, not a victim —
   and leaving it eligible means the injection achieves its goal (promotion)
   even though the model never obeyed it.

`ProductCandidate.trust_flagged` implements (3): the candidate is built and
counted, excluded from ranking by `is_recommendable`, refused again by the
recommendation validator as `SELLER_CONTENT_FLAGGED` (defence in depth), and the
exclusion is **disclosed to the user** rather than being a silent delisting.

The `FK-INJECT-01` fixture exercises this on every run. It is priced at
₹31,990 with 8GB RAM: cheap enough that on a query without a RAM floor it
out-scores every legitimate option (0.548 vs 0.512), which is exactly why the
gate is needed rather than relying on it failing on merit.

## 4. Trust framing

The rule: **untrusted text is never concatenated into a system prompt.** System
instructions come only from `prompts/`, which contains no request-derived data —
`test_caching.py::test_prefix_contains_no_request_specific_content` enforces it.

Untrusted content travels as message content inside a labelled, delimited block.
Three mechanisms make the delimiter hard to escape:

1. The fence token is a per-process random hex string, so it cannot be guessed
   from source.
2. Occurrences of the fence token inside the payload are stripped.
3. Payloads are length-capped, so a hostile description cannot flood context.

Marketplace data is additionally passed as **structured JSON**, so the model
reads named fields rather than a sentence that can pose as guidance.

## 5. Tool arguments

Every tool argument is a Pydantic model with `extra="forbid"` and closed bounds.
The important property is what these models make impossible: there is **no
field anywhere that accepts a URL, host, path, header or body**. A marketplace
client owns its base URL as a class attribute, so no model output can steer a
request. `test_tool_guardrails.py::test_search_request_has_no_url_field` asserts
this structurally rather than by convention.

Queries are reduced to a safe character set rather than escaped, because a laptop
search needs only alphanumerics and a little punctuation.

## 6. Tool responses

Raw provider JSON stops being raw here. A failing payload is **quarantined with
a reason**, never silently dropped — a marketplace quietly returning malformed
prices must be visible in the audit trail.

Checks: envelope shape, required identifiers, URL parses and uses HTTPS, host
belongs to the claimed marketplace, prices positive and coherent, discounts
non-negative and not larger than the listed price, no duplicate ids, currency
supported.

The host allowlist (`TRUSTED_HOSTS`) is per marketplace, and the sets are
disjoint — an Amazon URL does not validate for Flipkart. This is the only defence
against a compromised provider steering users off-platform.

## 7. Prices

No LLM participates in any part of `price_validator.py`. Checks:

- `price > 0`, `discount >= 0`, `discount <= listed price`, `effective price >= 0`
- currency supported and consistent throughout
- **no duplicate discounts** — both the same offer id twice, and the same
  `(kind, amount)` under two different ids, which is how a double discount
  actually happens in production data
- **conditional discounts are not assumed** — a bank offer, coupon or exchange
  bonus is applied only when the profile satisfies it; otherwise it is reported
  separately as `unmet_conditional_offers`
- **cashback is never subtracted** from the checkout price; it is tracked in its
  own field
- no-cost EMI changes financing terms, not price

Offers are processed in a deterministic order so the computation is reproducible
— that is what lets layer 9 recompute and compare. `PriceBreakdown` validates its
own arithmetic, so an incoherent breakdown cannot be constructed at all.

If price data is inconsistent, the product is marked invalid and never
recommended. An unusable price is never turned into an attractive one.

## 8. LLM price claims

The prompt tells the model not to state figures and the schema has no monetary
fields, but "the prompt said not to" is not a guardrail. `price_claims.py`
removes any monetary figure or percentage in model prose that does not match a
value the pricing code actually computed. Specification figures (`16GB`,
`14 inch`, `1.4 kg`) are exempted so they are not mangled.

An offending clause is replaced with neutral wording and the incident is audited,
rather than failing the whole request over a cosmetic slip.

## 9. Recommendation validator

The last gate, and deliberately independent of what produced the recommendation.
It verifies:

- the candidate exists in the candidate set
- the candidate was returned by a provider **in this run**
- the URL is byte-identical to the provider's URL and on a trusted host
- the title matches the provider's title, and the product is in stock
- hard requirements pass, re-evaluated from provider data
- the price is valid and **reproducible** — recomputed and compared field by field
- no mandatory constraint is violated, re-derived against the recomputed price
- the score is **reproducible** — recomputed with the same scoring function
- the recommendation is in the ranked set and is its highest scorer

On failure the graph **routes back** to `product_ranking` with the offending
candidate excluded, bounded by `recommendation_max_repair_attempts`. When
attempts are exhausted the agent returns no recommendation rather than an
unverified one. The LLM is never consulted about the verdict.

### Why provenance works without signing

`ProviderRegistry` is built only from validated provider responses, in-process,
per run. Product data is never carried across turns and never accepted from a
client, so membership is a genuine provenance proof rather than an assertion.
This is the reason the session store holds requirements only — see
`docs/OBSERVABILITY.md`.

## LLM output guardrails

All structured LLM responses are Pydantic models (`llm/schemas.py`). No schema
has a monetary field, and none carries the winner's identity — the model cannot
change which product is recommended or state a price.

Invocation (`llm/structured.py`) constrains output with a JSON schema, validates,
and on failure **retries once with a different constraining mechanism**
(`json_schema` → `function_calling`). Repeating an identical failing request is
rarely useful. If it still fails, the caller degrades gracefully. There is no
free-text fallback, and no `eval`, `exec` or dynamic execution anywhere.

## PII and secrets

`security/redaction.py` is the single redactor used by structured logging **and**
LangSmith tracing, so a pattern added once protects both. Redaction lives in the
log formatter rather than at call sites, so it cannot be bypassed by a log
statement written later.

Card numbers are **removed at the input boundary**, not merely masked at the log
boundary. `"My HDFC card ends in 1234"` becomes eligibility only:
`PurchaseProfile(eligible_banks=[Bank.HDFC])`. That model has no field capable of
holding a card number, uses `extra="forbid"`, and a validator rejects digit runs
so a future free-text field cannot quietly become somewhere numbers land.

Luhn validation prevents false positives — a 21-digit order number is not masked,
and `"16GB RAM under 80000 rupees"` passes through untouched.

## Testing

228 tests, all offline (no API key, no network). The security-relevant ones:

| File | What it proves |
|------|----------------|
| `test_input_guardrails.py` | 17 attack strings blocked; 7 legitimate requests not |
| `test_scope_guardrail.py` | Disallowed topics blocked at every stage; terse answers allowed |
| `test_marketplace_injection.py` | Hostile listing flagged, disqualified and disclosed; never wins even when cheapest; payload neutralised before display; fence cannot be escaped |
| `test_tool_guardrails.py` | No URL field; 13 malformed-product cases quarantined |
| `test_price_validator.py` | All eight price rules, plus order-independence |
| `test_recommendation_validator.py` | 15 tampering attacks caught |
| `test_repair_loop.py` | Failed validation routes back and recovers; loop is bounded |
| `test_redaction_and_logging.py` | Secrets masked, legitimate text untouched |
| `test_graph_flow.py` | Attacks never reach a marketplace; card digits never stored |
| `test_api.py` | 422 on malformed requests; 500 leaks nothing |
