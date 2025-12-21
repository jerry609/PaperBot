from __future__ import annotations

import logging
from typing import Iterable, Union, Optional

from paperbot.application.collaboration.message_schema import AgentEventEnvelope
from paperbot.application.ports.event_log_port import EventLogPort


class LoggingEventLog(EventLogPort):
    """
    Phase-0 event log: emit events as JSON lines to the Python logger.

    This gives immediate observability without introducing a DB or queue.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self._logger = logger or logging.getLogger("paperbot.eventlog")
        self._level = level

    def append(self, event: Union[AgentEventEnvelope, dict]) -> None:
        if isinstance(event, AgentEventEnvelope):
            payload = event.to_json()
        else:
            # best effort
            try:
                payload = AgentEventEnvelope(**event).to_json()  # type: ignore[arg-type]
            except Exception:
                payload = str(event)
        self._logger.log(self._level, payload)

    def stream(self, run_id: str) -> Iterable[dict]:
        # Logging backend cannot stream retrospectively.
        return iter(())

    def close(self) -> None:
        return None


