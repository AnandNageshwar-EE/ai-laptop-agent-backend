"""Evaluation runner.

    # free, offline, deterministic — safe in CI
    python -m laptop_agent.evals.run

    # only the adversarial rows, no pipeline runs
    python -m laptop_agent.evals.run --only adversarial

    # exercise the real model (costs money; keeps fixtures for comparability)
    python -m laptop_agent.evals.run --live

    # push the dataset and record an experiment in LangSmith
    python -m laptop_agent.evals.run --langsmith

Marketplace data is pinned to fixtures in every mode. Live listings would make
two runs incomparable — a "regression" could just be a price change — and the
point of an experiment is that the prompt or the model is the only variable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from ..agent import AgentReply, LaptopAgent
from ..config import Settings
from .dataset import CASES, EvalCase, cases_for
from .evaluators import BLOCKING_KEYS, EvalResult, run_all


def _reply_to_dict(reply: AgentReply) -> dict[str, Any]:
    """Shape the reply the way the evaluators (and the HTTP API) see it."""
    from ..api.routes import _to_view

    view = _to_view(reply.recommendation)
    return {
        "blocked": reply.blocked,
        "block_reason": reply.block_reason,
        "awaiting_clarification": reply.awaiting_clarification,
        "clarification_question": reply.clarification_question,
        "response_text": reply.response_text,
        "recommendation": view.model_dump(mode="json") if view else None,
        "diagnostics": reply.diagnostics,
    }


def build_settings(*, live: bool) -> Settings:
    return Settings(
        llm_mode="live" if live else "offline",
        # Pinned deliberately — see the module docstring.
        marketplace_source="fixtures",
        serpapi_key=None,
        langsmith_tracing=False,
        environment="eval",
    )


def run_case(agent: LaptopAgent, case: EvalCase) -> tuple[dict[str, Any], list[EvalResult]]:
    reply = agent.run(message=case.message)
    payload = _reply_to_dict(reply)
    return payload, run_all(case, payload)


def run_local(cases: tuple[EvalCase, ...], *, live: bool, verbose: bool) -> int:
    settings = build_settings(live=live)
    agent = LaptopAgent(settings=settings)

    print(f"laptop-agent evaluation · {len(cases)} cases")
    print(f"  llm_mode={settings.llm_mode}  marketplace={settings.marketplace_source}\n")

    per_check: dict[str, Counter[str]] = {}
    failed_rows: list[tuple[str, list[EvalResult]]] = []
    blocking_failures = 0

    for case in cases:
        payload, results = run_case(agent, case)
        bad = [r for r in results if r.applicable and not r.passed]
        for result in results:
            counter = per_check.setdefault(result.key, Counter())
            counter["skip" if not result.applicable else ("pass" if result.passed else "fail")] += 1

        blocking_bad = [r for r in bad if r.key in BLOCKING_KEYS]
        blocking_failures += len(blocking_bad)
        mark = "PASS" if not bad else ("FAIL" if blocking_bad else "warn")
        print(f"  [{mark}] {case.id:26} {case.expect.value}")
        if bad:
            failed_rows.append((case.id, bad))
            for result in bad:
                flag = "blocking" if result.key in BLOCKING_KEYS else "signal"
                print(f"           {flag:8} {result.key}: {result.comment}")
        if verbose and not bad:
            rec = payload.get("recommendation")
            if rec:
                print(f"           -> {rec['title'][:52]}  {rec['effective_price']}")

    print("\nper-check results")
    width = max(len(k) for k in per_check)
    for key, counter in per_check.items():
        scored = counter["pass"] + counter["fail"]
        rate = f"{counter['pass']}/{scored}" if scored else "—"
        tag = "blocking" if key in BLOCKING_KEYS else "signal  "
        print(f"  {tag} {key:{width}}  {rate:>7}   (skipped {counter['skip']})")

    print(f"\n{len(cases) - len(failed_rows)}/{len(cases)} cases fully clean")
    print(f"blocking check failures: {blocking_failures}")
    return 1 if blocking_failures else 0


def run_langsmith(cases: tuple[EvalCase, ...], *, live: bool) -> int:
    """Upload the dataset and record an experiment.

    Prompt and model versions ride along as experiment metadata, which is what
    makes "prompt v1 vs v2" a comparison rather than a guess (spec section 5.1).
    """
    try:
        from langsmith import Client, evaluate
    except ImportError:
        print("langsmith is not installed")
        return 2

    settings = build_settings(live=live)
    if settings.langsmith_api_key is None:
        # Settings deliberately disables tracing for evals; read the key directly.
        from ..config import get_settings

        settings = settings.model_copy(
            update={"langsmith_api_key": get_settings().langsmith_api_key}
        )
    if settings.langsmith_api_key is None:
        print("LANGSMITH_API_KEY is not set")
        return 2

    client = Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key.get_secret_value(),
    )
    name = "laptop-agent-golden"

    existing = list(client.list_datasets(dataset_name=name))
    dataset = existing[0] if existing else client.create_dataset(
        dataset_name=name,
        description="Invariant-based golden set: outcome, budget, specs, price integrity, guardrails.",
    )
    if not existing:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"message": c.message} for c in cases],
            outputs=[{"expect": c.expect.value, "intent": c.intent} for c in cases],
            metadata=[{"case_id": c.id, "adversarial": c.is_adversarial} for c in cases],
        )
        print(f"created dataset {name!r} with {len(cases)} examples")
    else:
        print(f"reusing dataset {name!r}")

    by_message = {c.message: c for c in cases}
    agent = LaptopAgent(settings=settings)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return _reply_to_dict(agent.run(message=inputs["message"]))

    def make_evaluator(index: int):
        from .evaluators import EVALUATORS

        evaluator = EVALUATORS[index]

        def wrapped(inputs: dict, outputs: dict, **_: Any) -> dict[str, Any]:
            case = by_message.get(inputs.get("message", ""))
            if case is None:
                return {"key": evaluator.__name__, "score": None, "comment": "unknown case"}
            result = evaluator(case, outputs or {})
            return {"key": result.key, "score": result.score, "comment": result.comment}

        wrapped.__name__ = evaluator.__name__
        return wrapped

    from .evaluators import EVALUATORS

    results = evaluate(
        target,
        data=client.list_examples(dataset_name=name),
        evaluators=[make_evaluator(i) for i in range(len(EVALUATORS))],
        experiment_prefix=f"{settings.llm_mode}-{settings.traced_model_name}",
        metadata={
            "llm_mode": settings.llm_mode,
            "model": settings.traced_model_name,
            "llm_provider": settings.llm_provider,
            "marketplace_source": settings.marketplace_source,
            "graph_version": settings.graph_version,
            **_prompt_versions(),
        },
        client=client,
        max_concurrency=2,
    )
    print(f"\nexperiment recorded: {getattr(results, 'experiment_name', 'see LangSmith')}")
    print(f"open {settings.langsmith_endpoint.replace('api.', '')} -> Datasets -> {name}")
    return 0


def _prompt_versions() -> dict[str, str]:
    from ..prompts.provider import PromptTask, get_prompt_provider

    versions = get_prompt_provider().versions(PromptTask.RECOMMENDATION)
    return {k: v for k, v in versions.items() if "version" in k or "fingerprint" in k}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the laptop-agent evaluation suite.")
    parser.add_argument("--only", choices=("all", "adversarial", "quality"), default="all")
    parser.add_argument("--live", action="store_true", help="call the real model (costs money)")
    parser.add_argument("--langsmith", action="store_true", help="record an experiment")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = cases_for(args.only)
    if args.langsmith:
        return run_langsmith(cases, live=args.live)
    return run_local(cases, live=args.live, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
