"""Shared audit helper for MCP tool calls.

Every tool call logs an AgentEventEnvelope to EventLogPort with
workflow='mcp', stage='tool_call'. If EventLogPort is not registered
in the DI Container, auditing degrades silently (no tool failure).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from paperbot.application.collaboration.message_schema import (
    make_event,
    new_run_id,
    new_trace_id,
)
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.core.di import Container

logger = logging.getLogger(__name__)


def _get_event_log() -> Optional[EventLogPort]:
    """Resolve EventLogPort from Container, return None on any exception."""
    try:
        return Container.instance().resolve(EventLogPort)
    except Exception:
        return None


def log_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    result_summary: str,
    duration_ms: float,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> str:
    """Log an MCP tool call as an AgentEventEnvelope.

    Args:
        tool_name: Name of the MCP tool being called.
        arguments: Arguments passed to the tool.
        result_summary: Short summary of the tool result.
        duration_ms: Duration of the tool call in milliseconds.
        run_id: Optional run_id for correlation. If None, a new one is generated.
        error: Optional error message if the tool call failed.

    Returns:
        The run_id used (provided or generated).
    """
    rid = run_id if run_id else new_run_id()

    event = make_event(
        run_id=rid,
        trace_id=new_trace_id(),
        workflow="mcp",
        stage="tool_call",
        attempt=0,
        agent_name="paperbot-mcp",
        role="system",
        type="error" if error is not None else "tool_result",
        payload={
            "tool": tool_name,
            "arguments": arguments,
            "result_summary": result_summary,
            "error": error,
        },
        metrics={"duration_ms": duration_ms},
    )

    event_log = _get_event_log()
    if event_log is not None:
        try:
            event_log.append(event)
        except Exception:
            logger.debug("Failed to append audit event for tool %s", tool_name, exc_info=True)

    return rid
