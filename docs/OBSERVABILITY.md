# Observability

## LangSmith tracing

### Configuration

Environment only. Nothing is hard-coded, and tracing is off unless both the flag
and a key are present.

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=          # never committed
LANGSMITH_PROJECT=ai-laptop-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

When disabled the graph runs unchanged. Tracer construction failures are caught
and logged — observability must never take the application down.

### Redaction happens inside the tracing client

The LangSmith client is constructed with `hide_inputs` and `hide_outputs` bound to
the same `SecretRedactor` the logs use:

```python
Client(
    api_url=...,
    api_key=...,
    hide_inputs=redactor.hide_inputs,
    hide_outputs=redactor.hide_outputs,
)
```

Every payload is filtered before leaving the process, and a pattern added to the
redactor protects logs and traces together.

The tracer is passed explicitly in the run config rather than enabled through
ambient environment side effects, so a process tracing this graph does not
accidentally trace everything else it does.

### Run structure

Top-level run name: **`laptop_agent_session`**. Nested runs are the LangGraph node
names, so the graph structure and the trace structure cannot drift apart:

```
laptop_agent_session
├── input_guardrail
├── requirements_analysis
├── clarification_decision
├── search_planning
├── amazon_search          ─┐ concurrent
├── flipkart_search        ─┘
├── offer_analysis
├── pricing_calculation
├── product_ranking
├── recommendation_generation
└── recommendation_validation
```

`product_ranking` onward may appear more than once when the recommendation
validator rejects a candidate and the graph routes back.

### What is captured

Inputs needed for debugging, structured outputs, tool names, metadata, latency,
errors, token usage and retry count.

**Hidden chain-of-thought is not captured.** The configured model returns thinking
with `display: "omitted"` by default and nothing here requests summarised
reasoning.

### Metadata

Static, on every run:

```json
{
  "application": "ai-laptop-agent",
  "environment": "local",
  "graph_version": "v1",
  "model": "claude-opus-5",
  "llm_provider": "anthropic"
}
```

Per session, from `RunMetrics.as_trace_metadata()` plus the agent's diagnostics:

| Field | Meaning |
|-------|---------|
| `session_id` | Opaque, non-guessable, carries no user information |
| `user_requirement_category` | The extracted use case |
| `marketplace_provider_count`, `marketplaces_used` | How many providers answered |
| `candidate_count` | Priced, constraint-checked candidates |
| `selected_candidate` | `marketplace:product_id` of the winner |
| `clarification_required` | Whether the agent asked a question |
| `trade_off_required` | Whether no candidate met every preference |
| `total_latency_ms`, `node_latencies_ms` | Overall and per-node timing |
| `llm_calls`, `retry_count` | Model call count and structured-output retries |
| `input_tokens`, `output_tokens` | Token usage where available |
| `cache_read_input_tokens`, `cache_creation_input_tokens`, `prompt_cache_hit` | Provider-side prompt cache effectiveness |
| `product_cache_hits`, `product_cache_misses` | Volatile provider cache |
| `products_returned`, `products_quarantined`, `offers_quarantined` | Tool-output guardrail activity |
| `injection_flags` | Listings whose text attempted an injection |
| `guardrail_blocks` | Refused turns |
| `recommendation_validation_attempts`, `recommendation_validation_failures` | Validator activity |
| `prompt_versions`, `prompt_prefix_fingerprint` | Which prompt produced this run |

No payment information is recorded. Card eligibility appears only as a boolean.

### Tags

Applied consistently so traces can be filtered:

```
agent:laptop-shopping
graph:v1
environment:local
marketplace:amazon
marketplace:flipkart
```

## Structured logging

JSON to stdout, one object per record, redacted in the formatter. Noisy HTTP
libraries are pinned to WARNING. On an exception only the type and message are
recorded — never a traceback, which can carry local variable values.

## Audit trail

Guardrail decisions, price validations and recommendation verdicts are emitted as
an append-only stream of `AuditRecord`s (`audit/sink.py`). Events:

```
input_blocked            scope_rejected           injection_detected
tool_args_rejected       product_quarantined      offer_quarantined
price_invalid            llm_output_invalid       price_claim_stripped
recommendation_rejected  recommendation_approved
```

Attack and integrity events log at WARNING; approvals at INFO. Records name a
*category* and never the matched span, so attacker-controlled text does not land
in logs.

`AuditSink` is a Protocol. `StructuredLogAuditSink` is the default;
`CollectingAuditSink` is the test double.

### Why the audit trail is a log stream, not a database table

This data is an append-only event stream, not transactional state. A Postgres
audit table would reimplement retention, rotation and indexing, couple request
latency to a database write, and duplicate what LangSmith already stores per run.
Structured logs go to whatever collector the platform already runs.

## Persistence: why there is no database

The backend is **stateless with respect to product data**. Candidates, prices and
offers are re-fetched every run and never persist across a turn — required
anyway, since prices are volatile, and it is what makes the recommendation
validator's provenance check sound (see `docs/GUARDRAILS.md`).

What survives a turn is small and user-owned: the requirements gathered so far
and the offer-eligibility profile. That lives in `session/store.py`, TTL-bounded
and size-capped, holding no prices and no PII.

The trust boundary this creates:

- **Requirements** — user-owned; the user is allowed to change them; re-validated
  every turn. Safe to carry.
- **Products, prices, URLs, scores** — provider-owned; never accepted from a
  client; always re-fetched in-process.

Not having a database also means there is no store of user requirements to
encrypt, back up, retain, or answer deletion requests about.

`SessionStore` and `AuditSink` are Protocols, so a durable implementation drops in
without touching graph or node code. Concrete triggers that would justify one:

- resuming a session hours later on another device
- saved comparisons or price-drop watches (durable background jobs)
- immutable, queryable audit records with retention guarantees logs cannot meet
- idempotency keys, once real purchases exist

None are in scope today.

## Health endpoint

`GET /health` reports the configuration facts worth confirming at a glance,
including `prompt_prefix_fingerprint` — so a deployment that accidentally changed
the cached prefix is visible here rather than only as a collapsed cache hit rate.
