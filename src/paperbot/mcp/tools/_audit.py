"""Shared audit helper for MCP tool calls.

Every tool call logs an AgentEventEnvelope to EventLogPort with
workflow='mcp', stage='tool_call'. If EventLogPort is not registered
in the DI Container, auditing degrades silently (no tool failure).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Union

from paperbot.application.collaboration.message_schema import (
    EventType,
    make_event,
    new_run_id,
    new_trace_id,
)
from paperbot.application.ports.event_log_port import EventLogPort
from paperbot.core.di import Container

logger = logging.getLogger(__name__)

_SENSITIVE_KEY_PARTS = frozenset({"api_key", "token", "password", "secret", "authorization"})
_MAX_AUDIT_TEXT_LENGTH = 1000
_MAX_AUDIT_COLLECTION_ITEMS = 20
_MAX_AUDIT_NESTING_DEPTH = 4


def _truncate_text(value: str) -> str:
    text = str(value)
    if len(text) <= _MAX_AUDIT_TEXT_LENGTH:
        return text
    return text[:_MAX_AUDIT_TEXT_LENGTH] + "...[truncated]"


def _truncate_json(value: Any) -> str:
    try:
        return _truncate_text(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return _truncate_text(str(value))


def _is_sensitive_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _sanitize_value(value: Any, *, key: Optional[str] = None, depth: int = 0) -> Any:
    if key and _is_sensitive_key(key):
        return "***redacted***"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_text(value)

    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for index, (child_key, child_value) in enumerate(value.items()):
            if index >= _MAX_AUDIT_COLLECTION_ITEMS:
                sanitized["__truncated__"] = True
                break
            normalized_key = str(child_key)
            sanitized[normalized_key] = _sanitize_value(
                child_value,
                key=normalized_key,
                depth=depth + 1,
            )
        if depth >= _MAX_AUDIT_NESTING_DEPTH:
            return _truncate_json(sanitized)
        return sanitized

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized_items = [
            _sanitize_value(item, depth=depth + 1) for item in items[:_MAX_AUDIT_COLLECTION_ITEMS]
        ]
        if len(items) > _MAX_AUDIT_COLLECTION_ITEMS:
            sanitized_items.append("...[truncated]")
        if depth >= _MAX_AUDIT_NESTING_DEPTH:
            return _truncate_json(sanitized_items)
        return sanitized_items

    if depth >= 2:
        return _truncate_json(value)

    return _truncate_text(str(value))


def _sanitize_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): _sanitize_value(value, key=str(key)) for key, value in arguments.items()}


def _get_event_log() -> Optional[EventLogPort]:
    """Resolve EventLogPort from Container, return None on any exception."""
    try:
        return Container.instance().resolve(EventLogPort)
    except Exception:
        return None


def log_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    result_summary: Union[str, Dict[str, Any]],
    duration_ms: float,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> str:
    """Log an MCP tool call as an AgentEventEnvelope.

    Args:
        tool_name: Name of the MCP tool being called.
        arguments: Arguments passed to the tool.
        result_summary: Short summary of the tool result. May be a string or structured dict.
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
        type=EventType.TOOL_ERROR if error is not None else EventType.TOOL_RESULT,
        payload={
            "tool": tool_name,
            "arguments": _sanitize_arguments(arguments),
            "result_summary": _sanitize_value(result_summary),
            "error": _truncate_text(error) if error is not None else None,
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
