from __future__ import annotations

import logging
from typing import Iterable, List, Union

from paperbot.application.collaboration.message_schema import AgentEventEnvelope
from paperbot.application.ports.event_log_port import EventLogPort

logger = logging.getLogger(__name__)


class CompositeEventLog(EventLogPort):
    """
    Tee events to multiple backends.

    Phase 1A: used to keep JSONL logging while persisting to SQLite.
    """

    def __init__(self, backends: List[EventLogPort]):
        self._backends = [b for b in backends if b is not None]

    def append(self, event: Union[AgentEventEnvelope, dict]) -> None:
        for backend in self._backends:
            try:
                backend.append(event)
            except Exception as e:
                logger.debug(f"CompositeEventLog backend append failed: {e}")

    def stream(self, run_id: str) -> Iterable[dict]:
        # Prefer the first backend that supports streaming.
        for backend in self._backends:
            try:
                it = backend.stream(run_id)
                # materialize one element to see if it's usable
                first = next(iter(it), None)
                if first is None:
                    continue
                # re-yield first then the rest (best-effort)
                def _gen():
                    yield first
                    for x in it:
                        yield x

                return _gen()
            except Exception:
                continue
        return iter(())

    def close(self) -> None:
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                pass

    # ---- Optional convenience APIs (used by replay routes) ----

    def list_runs(self, limit: int = 50):
        for backend in self._backends:
            if hasattr(backend, "list_runs"):
                try:
                    return backend.list_runs(limit=limit)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.debug(f"CompositeEventLog backend list_runs failed: {e}")
        return []

    def list_events(self, run_id: str, *, trace_id=None, limit: int = 1000):
        for backend in self._backends:
            if hasattr(backend, "list_events"):
                try:
                    return backend.list_events(run_id, trace_id=trace_id, limit=limit)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.debug(f"CompositeEventLog backend list_events failed: {e}")
        return []


