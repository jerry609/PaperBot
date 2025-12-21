from __future__ import annotations

from typing import Protocol, runtime_checkable, Iterable, Optional, Union, Any

from paperbot.application.collaboration.message_schema import AgentEventEnvelope


@runtime_checkable
class EventLogPort(Protocol):
    """
    Minimal event log port.

    Phase-0 goal: enable event emission without choosing persistence yet.
    Implementations may log to stdout, write JSONL files, or persist to DB.
    """

    def append(self, event: Union[AgentEventEnvelope, dict]) -> None:
        """Append an event (envelope)."""

    def stream(self, run_id: str) -> Iterable[dict]:
        """
        Stream events by run_id.

        Phase-0: optional / may return empty iterator depending on implementation.
        """

    def close(self) -> None:
        """Close underlying resources (optional)."""


