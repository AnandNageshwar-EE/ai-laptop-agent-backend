# AI Laptop Shopping Agent — Backend

A LangGraph agent that takes a laptop shopping request, searches multiple
marketplaces, validates every price, and recommends one laptop it can justify —
with production-grade guardrails, prompt caching and LangSmith tracing.

The Streamlit UI is a **separate service** in its own repository:
[ai-laptop-agent-frontend](https://github.com/AnandNageshwar-EE/ai-laptop-agent-frontend).
It talks to this backend over HTTP only and shares no code with it.

## Quick start

Runs with no API key and no network — `LLM_MODE=offline` uses a deterministic
rule-based reasoner against the same schemas, so the entire pipeline is
exercisable out of the box.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env

.venv/bin/python -m pytest -q                 # 228 tests, all offline
.venv/bin/python -m uvicorn laptop_agent.api.main:app --reload
```

```bash
curl localhost:8000/health

curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"laptop for software development under 80000 with at least 16GB RAM"}'
```

To use the model instead of the offline reasoner:

```bash
LLM_MODE=live ANTHROPIC_API_KEY=sk-ant-... \
  .venv/bin/python -m uvicorn laptop_agent.api.main:app
```

## What it does

```
input_guardrail
    ├─ blocked ──────────────────────────────────────────► END (safe refusal)
    └─ requirements_analysis
           └─ clarification_decision
                  ├─ needs clarification ────────────────► END (awaits answer)
                  └─ search_planning
                         ├─► amazon_search   ─┐ concurrent
                         └─► flipkart_search ─┘
                                └─ offer_analysis
                                      └─ pricing_calculation
                                            └─ product_ranking
                                                  ├─ no candidates ──► END
                                                  └─ recommendation_generation
                                                        └─ recommendation_validation
                                                              ├─ valid ─────► END
                                                              ├─ retry ─────► product_ranking
                                                              └─ exhausted ─► END
```

The validator is the point of the design. It re-derives the price, the hard
constraints and the score from provider data and compares them against the
recommendation. On mismatch the graph routes back and re-ranks with the offending
candidate excluded. The LLM cannot override it.

## Design decisions worth knowing

**No database.** The backend is stateless with respect to product data — prices
are re-fetched every run, which is required anyway because they are volatile, and
is what makes the validator's provenance check sound. Session state is
requirements plus offer eligibility, held in memory. `SessionStore` and
`AuditSink` are Protocols so a durable implementation drops in later.
Reasoning and the triggers that would justify one: [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).

**No LLM touches a price.** Every monetary value originates in structured
marketplace data and flows through `PriceValidator`. No LLM output schema has a
monetary field, and model prose is screened for invented figures.

**Marketplace text is data, never instruction.** A hostile product description is
kept and flagged rather than rejected — rejecting it would let a competitor
delist a rival by poisoning its description.

**Deterministic where determinism matters.** Guardrails, pricing and ranking are
pure functions. The model extracts requirements and writes prose; it does not
decide, price or rank.

## Layout

```
src/laptop_agent/
├── config.py          Settings (env only, secrets as SecretStr)
├── agent.py           Public entry point
├── domain/            Validated models — nothing enters state except these
├── guardrails/        Nine independent layers  → docs/GUARDRAILS.md
├── prompts/           Versioned, stable prefixes → docs/PROMPT_CACHING.md
├── llm/               Schemas, structured invocation, offline reasoner
├── marketplace/       Clients (own their base URLs) + fixtures
├── pricing/           Deterministic candidate assembly
├── ranking/           Deterministic, reproducible scoring
├── observability/     LangSmith tracing + per-run metrics
├── session/           In-memory session state (no prices, no PII)
├── audit/             Append-only decision trail
├── security/          Redaction + structured logging
├── graph/             Typed state, nodes, builder
└── api/               FastAPI app
```

## Marketplace data

Amazon and Flipkart are simulated from `marketplace/fixtures.py` so the pipeline
runs without credentials. Going live means replacing two methods
(`_fetch_products`, `_fetch_offers`) with HTTP calls — nothing downstream changes,
because everything downstream already treats the response as untrusted.

The fixtures deliberately include hostile and malformed records, so the guardrails
are exercised by the default demo rather than only by tests:

| Fixture | Exercises |
|---------|-----------|
| `FK-INJECT-01` | Prompt injection in a seller description |
| `AMZ-BADPRICE-1` | Negative price |
| `FK-BADURL-1` | URL pointing off-marketplace |
| `AMZ-NOID-1` | Missing product identifier |
| `FK-FAKEMRP-1` | Inflated MRP faking a 97% discount |
| `AMZ-OFF-G16-UPFRONT-DUP` | Duplicate discount record |
| `AMZ-OFF-IP5-CASHBACK` | Cashback that must not reduce checkout price |
| `AMZ-OFF-IP5-BANK` | Conditional HDFC offer that must not be assumed |
| `AMZ-OFF-ORPHAN` | Offer for a product never returned |
| `FK-OFF-IMPOSSIBLE` | Discount larger than the listed price |

A default run quarantines 4 listings and 2 offers, and flags 1 injection attempt.

## Documentation

- [docs/GUARDRAILS.md](docs/GUARDRAILS.md) — the nine layers, what each prevents
- [docs/PROMPT_CACHING.md](docs/PROMPT_CACHING.md) — provider-side vs local caching, and the difference
- [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) — tracing, metadata, tags, and why there is no database

## Configuration

See [.env.example](.env.example). Notable constraints enforced in code:

- `PRODUCT_CACHE_TTL_SECONDS` and `OFFER_CACHE_TTL_SECONDS` are **rejected above
  300** — prices are volatile by policy, and the cache also clamps at `set()`.
- `CORS_ALLOW_ORIGINS` is an explicit allowlist; `*` is not used.
- API keys are `SecretStr`, so an accidental f-string yields `**********`.

## Docker

```bash
docker compose up --build          # backend on :8000, frontend on :8501
```

The `frontend` service expects the frontend repository checked out as a sibling
directory. Comment it out to run the backend alone.

## Tests

```bash
.venv/bin/python -m pytest -q                                   # all 228
.venv/bin/python -m pytest tests/test_recommendation_validator.py -v
```

No API key, no network, no mocking framework — offline mode is a real code path,
not a test fixture.
