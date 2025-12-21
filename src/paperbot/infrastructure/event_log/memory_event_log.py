from __future__ import annotations

from typing import Iterable, Union, Dict, Any, List

from paperbot.application.collaboration.message_schema import AgentEventEnvelope
from paperbot.application.ports.event_log_port import EventLogPort


class InMemoryEventLog(EventLogPort):
    """Simple in-memory event log (useful for tests/evals)."""

    def __init__(self) -> None:
        self.events: List[dict] = []

    def append(self, event: Union[AgentEventEnvelope, dict]) -> None:
        if isinstance(event, AgentEventEnvelope):
            self.events.append(event.to_dict())
        else:
            self.events.append(dict(event))

    def stream(self, run_id: str) -> Iterable[dict]:
        return (e for e in self.events if e.get("run_id") == run_id)

    def close(self) -> None:
        return None


