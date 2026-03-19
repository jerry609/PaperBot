"""Agent event helper functions.

Convenience wrappers that produce correctly-typed AgentEventEnvelope instances
for the two most common event families in the PaperBot multi-agent system:

* Lifecycle events  — when an agent starts, works, completes, or errors
* Tool-call events  — when an MCP tool is invoked (success or error)

Both helpers delegate to ``make_event()`` so the underlying envelope structure
is always consistent with the rest of the codebase.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from paperbot.application.collaboration.message_schema import (
    AgentEventEnvelope,
    EventType,
    make_event,
    new_run_id,
    new_trace_id,
)


def make_lifecycle_event(
    *,
    status: str,
    agent_name: str,
    run_id: str,
    trace_id: str,
    workflow: str,
    stage: str,
    attempt: int = 0,
    role: str = "worker",
    detail: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> AgentEventEnvelope:
    """Build a lifecycle AgentEventEnvelope.

    Args:
        status: One of the ``EventType.AGENT_*`` constants.
        agent_name: Human-readable name of the agent emitting the event.
        run_id: Run correlation identifier.
        trace_id: Trace correlation identifier.
        workflow: Workflow name (e.g. "scholar_pipeline").
        stage: Stage name within the workflow.
        attempt: Retry attempt counter (default 0).
        role: Actor role — "orchestrator" / "worker" / "evaluator" / "system".
        detail: Optional free-text detail appended to the payload.
        metrics: Optional metrics dict forwarded to the envelope.
        tags: Optional tags dict forwarded to the envelope.

    Returns:
        AgentEventEnvelope with ``type=status`` and a standardised payload.
    """
    payload: Dict[str, Any] = {
        "status": status,
        "agent_name": agent_name,
    }
    if detail is not None:
        payload["detail"] = detail

    return make_event(
        run_id=run_id,
        trace_id=trace_id,
        workflow=workflow,
        stage=stage,
        attempt=attempt,
        agent_name=agent_name,
        role=role,
        type=status,
        payload=payload,
        metrics=metrics,
        tags=tags,
    )


def make_tool_call_event(
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    result_summary: Any,
    duration_ms: float,
    run_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    workflow: str = "mcp",
    stage: str = "tool_call",
    agent_name: str = "paperbot-mcp",
    role: str = "system",
    error: Optional[str] = None,
) -> AgentEventEnvelope:
    """Build a tool-call AgentEventEnvelope.

    Args:
        tool_name: Name of the MCP tool that was called.
        arguments: Arguments that were passed to the tool.
        result_summary: Short summary of the result (string or dict).
        duration_ms: Wall-clock duration of the call in milliseconds.
        run_id: Run correlation ID. Auto-generated if not provided.
        trace_id: Trace correlation ID. Auto-generated if not provided.
        workflow: Workflow name (default "mcp").
        stage: Stage name (default "tool_call").
        agent_name: Emitting agent name (default "paperbot-mcp").
        role: Actor role (default "system").
        error: Optional error message; presence flips type to ``TOOL_ERROR``.

    Returns:
        AgentEventEnvelope with ``type=TOOL_RESULT`` or ``type=TOOL_ERROR``.
    """
    event_type = EventType.TOOL_ERROR if error is not None else EventType.TOOL_RESULT

    return make_event(
        run_id=run_id if run_id is not None else new_run_id(),
        trace_id=trace_id if trace_id is not None else new_trace_id(),
        workflow=workflow,
        stage=stage,
        attempt=0,
        agent_name=agent_name,
        role=role,
        type=event_type,
        payload={
            "tool": tool_name,
            "arguments": arguments,
            "result_summary": result_summary,
            "error": error,
        },
        metrics={"duration_ms": duration_ms},
    )
