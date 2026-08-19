"""Evaluation: a golden dataset plus deterministic evaluators.

Distinct from the test suite. Tests assert that the code behaves correctly on
fixed inputs; evaluation asks whether the agent's *output* holds its invariants
across a realistic spread of requests, and lets two prompt or model versions be
compared on identical ground.
"""

from .dataset import CASES, EvalCase, Outcome, cases_for
from .evaluators import BLOCKING_KEYS, EVALUATORS, EvalResult, run_all

__all__ = [
    "BLOCKING_KEYS",
    "CASES",
    "EVALUATORS",
    "EvalCase",
    "EvalResult",
    "Outcome",
    "cases_for",
    "run_all",
]
