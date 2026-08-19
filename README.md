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

Two sources, selected by `MARKETPLACE_SOURCE`. Both return the same untrusted
envelope, so guardrails, pricing, ranking and validation are identical either way.

### `serpapi` — live listings, real prices, real product pages

```bash
MARKETPLACE_SOURCE=serpapi
SERPAPI_KEY=...            # serpapi.com, free tier = 250 searches/month
```

| Marketplace | Engine | Result |
|---|---|---|
| Amazon.in | `amazon` | Real ASINs, real prices, canonical `/dp/<ASIN>` URLs |
| Flipkart | `google` + `site:flipkart.com` | Real `/<slug>/p/itm…` URLs, price parsed from the snippet |

Three findings from building this, since the obvious approaches do not work:

- **`google_shopping` is a dead end for Flipkart.** It returns Flipkart rows with
  correct prices, but the only link is a Google catalog page — there is no
  `flipkart.com` URL for the provenance check to verify, so the rows are unusable.
  Organic search with a `site:` filter gives real product URLs instead.
- **Amazon result links are not product links.** They carry the search session
  (`/ref=sr_1_17?dib=…`), and sponsored rows are `/sspa/click` ad redirects. Both
  pass a host check while pointing at a tracker, so URLs are rebuilt from the ASIN.
- **Search APIs return no offer structures.** With a live source there are simply
  no offers rather than invented ones, so the effective price equals the real
  listed price. A discount that cannot be verified must never be applied.

Coverage is honestly asymmetric: Amazon has a dedicated engine, Flipkart does not,
so Amazon returns more results. Flipkart rows whose price cannot be parsed are
dropped, because a listing with no establishable price is useless to an agent
whose job is comparing what you would pay.

Specs are not returned by any search API, so `marketplace/spec_parser.py` extracts
them from listing titles by regex (`"(16GB/512GB SSD/Windows 11)"`). Anything it
cannot extract stays `None`, which the constraint logic treats as **unknown and
therefore failing** any mandatory requirement — claiming a laptop is under 1.5 kg
when no weight was reported would invent a fact the user is relying on.

### `fixtures` — simulated, offline, the default

No credentials, no network, deterministic. Prices are invented and will not match
the real marketplace; the UI says so. Going fully live means only swapping the
transport — nothing downstream changes, because everything downstream already
treats the response as untrusted.

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

## LLM providers and what they cost

`LLM_MODE=offline` (default) calls no model at all: `llm/offline.py` is a
deterministic rule-based reasoner producing the same schemas, so the whole
pipeline runs for free with no key. `LLM_MODE=live` routes through
`LLM_PROVIDER`:

| Provider | Endpoint shape | Prompt caching | Notes |
|---|---|---|---|
| `anthropic` | Anthropic Messages API | Native `cache_control` | Exact usage fields |
| `openrouter` | OpenAI chat-completions | Works for `anthropic/*`, `google/*`, `openai/*` | One key, many models |
| `bedrock` | Bedrock Converse | Provider-dependent | Needs `langchain-aws` |

OpenRouter is **OpenAI-compatible only** — there is no Anthropic-native
`/v1/messages` endpoint — so that path uses `ChatOpenAI` with an overridden base
URL. Two consequences that are handled rather than assumed:

- **Cache counters are named differently.** OpenRouter reports
  `prompt_tokens_details.cached_tokens`, not Anthropic's
  `input_token_details.cache_read`. Both shapes are parsed; without that the
  cache would appear to never hit and the stable prefix would be unverifiable.
- **`response_format: json_schema` is forwarded but not strictly enforced.**
  Measured against `anthropic/claude-opus-5` via OpenRouter, the json_schema
  attempt returned an invented shape (`budget:extra_forbidden`,
  `trade_offs.0:model_type`) and only the tool-calling retry succeeded. So
  gateway providers try **tool-calling first** (`strategies_for()`), which
  removes a guaranteed wasted round trip.

A `:free` model costs nothing but has no server-side prompt cache and follows
JSON schemas less reliably. `GET /health` reports
`provider_prompt_caching: false` in that case rather than implying token savings
that are not happening.

### Measured cost

Not estimated — measured over live runs with `anthropic/claude-opus-5` through
OpenRouter, at $5/MTok input, $25/MTok output, $0.50/MTok cache read:

| | Per query (2 LLM calls) |
|---|---|
| Input | ~10,300 tok, of which ~6,800 served from cache |
| Output | ~1,250 tok |
| **Cost** | **~$0.05–0.10** |

**Output dominates at ~60% of spend**, because Opus 5 runs adaptive thinking and
reasoning tokens bill at the output rate. Levers, in order of effect:

1. `OPENROUTER_MODEL=anthropic/claude-sonnet-5` — $2/$10 per MTok, roughly 2.5×
   cheaper, and noticeably faster.
2. Send fewer runner-ups to the explanation call. Each candidate costs ~257
   input tokens, so 5 runner-ups is ~1,300 tokens of the fresh input.
3. Prompt caching already removes ~66% of input tokens from full price; the
   stable-prefix work is what makes that hold.

Latency is 19–23 s per query on Opus 5 (two sequential calls, adaptive thinking).
Sonnet 5 is materially quicker.

### What live mode actually changes

The model handles vague-language interpretation ("something light", "won't die on
me") and writes better explanation prose. It does **not** touch prices, ranking
or the selection — those stay deterministic, and the recommendation validator
would reject its output if it tried. One consequence worth knowing: because
requirement *extraction* is the only nondeterministic step, an identical query
can yield a different winner between runs when the request contains fuzzy
language, since a slightly different extracted constraint changes which
candidates qualify. Offline mode is byte-identical run to run.

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

`tests/conftest.py` force-isolates every test onto `MARKETPLACE_SOURCE=fixtures`
with credentials cleared. Without that, a developer whose `.env` selects a live
source makes the suite slow, non-deterministic and quietly expensive —
`test_serpapi_transport.py` has explicit guards that fail if this regresses,
including one that treats any outbound HTTP call during a run as a failure.
