"""Session state.

The backend is stateless with respect to *product* data: candidates, prices and
offers are re-fetched on every run and never persist across a turn, because the
recommendation validator must be able to prove that a candidate came from a
provider in *this* run.

What does survive a turn is small and user-owned: the requirements gathered so
far and the purchase eligibility profile. That is what a session holds.

Storage is in-process and TTL-bounded. :class:`SessionStore` is a Protocol so a
durable implementation (Postgres, Redis) can be dropped in without touching
graph or node code — see ``docs/OBSERVABILITY.md`` for the trigger conditions
that would justify one.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..domain import LaptopRequirements, PurchaseProfile


class SessionState(BaseModel):
    """Everything carried between turns. Contains no payment data and no prices."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    requirements: LaptopRequirements | None = None
    purchase_profile: PurchaseProfile = Field(default_factory=PurchaseProfile)
    #: Set when the agent asked a clarifying question and is awaiting an answer.
    pending_clarification: str | None = None
    #: Redacted user turns, for context only. Never raw input.
    turn_count: int = 0

    def with_turn(self) -> SessionState:
        return self.model_copy(update={"turn_count": self.turn_count + 1})


@runtime_checkable
class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState | None: ...

    def put(self, state: SessionState) -> None: ...

    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """TTL-bounded, thread-safe, size-capped session store."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 1_800,
        max_sessions: int = 1_000,
        clock: Any = time.monotonic,
    ) -> None:
        self._data: dict[str, tuple[SessionState, float]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._clock = clock

    def get(self, session_id: str) -> SessionState | None:
        now = self._clock()
        with self._lock:
            found = self._data.get(session_id)
            if found is None:
                return None
            state, expires_at = found
            if expires_at <= now:
                del self._data[session_id]
                return None
            return state

    def put(self, state: SessionState) -> None:
        with self._lock:
            if len(self._data) >= self._max and state.session_id not in self._data:
                self._evict_locked()
            self._data[state.session_id] = (state, self._clock() + self._ttl)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def _evict_locked(self) -> None:
        now = self._clock()
        expired = [k for k, (_, exp) in self._data.items() if exp <= now]
        if expired:
            for key in expired:
                del self._data[key]
            return
        oldest = min(self._data, key=lambda k: self._data[k][1])
        del self._data[oldest]

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def new_session_id() -> str:
    """Opaque, non-guessable session identifier. Carries no user information."""
    return f"sess_{uuid.uuid4().hex}"
