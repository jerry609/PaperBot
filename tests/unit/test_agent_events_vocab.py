"""Unit tests for EventType constants and agent event helpers.

TDD: These tests define the required API. They will fail (RED) until
EventType, make_lifecycle_event, and make_tool_call_event are implemented.
"""

from __future__ import annotations

import pathlib


def test_event_type_constants():
    """All EventType constants are non-empty unique strings with the required sets."""
    from paperbot.application.collaboration.message_schema import EventType

    lifecycle_set = {
        EventType.AGENT_STARTED,
        EventType.AGENT_WORKING,
        EventType.AGENT_COMPLETED,
        EventType.AGENT_ERROR,
    }
    tool_set = {
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.TOOL_ERROR,
    }

    # All constants must be non-empty strings
    for name in (list(lifecycle_set) + list(tool_set)):
        assert isinstance(name, str), f"Expected str, got {type(name)}"
        assert name, "Expected non-empty string"

    # All constants must be unique across both sets
    all_constants = lifecycle_set | tool_set
    assert len(all_constants) == 7, "Expected 7 unique constants"

    # Required lifecycle values
    assert EventType.AGENT_STARTED == "agent_started"
    assert EventType.AGENT_WORKING == "agent_working"
    assert EventType.AGENT_COMPLETED == "agent_completed"
    assert EventType.AGENT_ERROR == "agent_error"

    # Required tool values
    assert EventType.TOOL_CALL == "tool_call"
    assert EventType.TOOL_RESULT == "tool_result"
    assert EventType.TOOL_ERROR == "tool_error"


def test_lifecycle_event_types():
    """make_lifecycle_event returns AgentEventEnvelope with correct type and payload keys."""
    from paperbot.application.collaboration.agent_events import make_lifecycle_event
    from paperbot.application.collaboration.message_schema import (
        AgentEventEnvelope,
        EventType,
    )

    run_id = "run-abc"
    trace_id = "trace-123"
    envelope = make_lifecycle_event(
        status=EventType.AGENT_STARTED,
        agent_name="test-agent",
        run_id=run_id,
        trace_id=trace_id,
        workflow="test-workflow",
        stage="test-stage",
    )

    assert isinstance(envelope, AgentEventEnvelope)
    assert envelope.type == "agent_started"
    assert envelope.run_id == run_id
    assert envelope.trace_id == trace_id
    assert "status" in envelope.payload
    assert "agent_name" in envelope.payload
    assert envelope.payload["agent_name"] == "test-agent"
    assert envelope.payload["status"] == EventType.AGENT_STARTED


def test_lifecycle_event_all_statuses():
    """Each lifecycle status produces an envelope with matching type field."""
    from paperbot.application.collaboration.agent_events import make_lifecycle_event
    from paperbot.application.collaboration.message_schema import EventType

    statuses = [
        EventType.AGENT_STARTED,
        EventType.AGENT_WORKING,
        EventType.AGENT_COMPLETED,
        EventType.AGENT_ERROR,
    ]

    for status in statuses:
        envelope = make_lifecycle_event(
            status=status,
            agent_name="test-agent",
            run_id="run-xyz",
            trace_id="trace-xyz",
            workflow="wf",
            stage="st",
        )
        assert envelope.type == status, f"Expected type={status!r}, got {envelope.type!r}"


def test_tool_call_event_success():
    """make_tool_call_event with no error returns envelope with type=tool_result and correct payload."""
    from paperbot.application.collaboration.agent_events import make_tool_call_event
    from paperbot.application.collaboration.message_schema import (
        AgentEventEnvelope,
        EventType,
    )

    envelope = make_tool_call_event(
        tool_name="paper_search",
        arguments={"query": "transformers"},
        result_summary="Found 5 papers",
        duration_ms=123.4,
        run_id="run-1",
        trace_id="trace-1",
    )

    assert isinstance(envelope, AgentEventEnvelope)
    assert envelope.type == EventType.TOOL_RESULT
    assert envelope.payload["tool"] == "paper_search"
    assert envelope.payload["arguments"] == {"query": "transformers"}
    assert envelope.payload["result_summary"] == "Found 5 papers"
    assert envelope.payload["error"] is None


def test_tool_call_event_error_type():
    """make_tool_call_event with error returns envelope with type=tool_error."""
    from paperbot.application.collaboration.agent_events import make_tool_call_event
    from paperbot.application.collaboration.message_schema import EventType

    envelope = make_tool_call_event(
        tool_name="paper_search",
        arguments={},
        result_summary="",
        duration_ms=10.0,
        run_id="run-2",
        trace_id="trace-2",
        error="boom",
    )

    assert envelope.type == EventType.TOOL_ERROR
    assert envelope.payload["error"] == "boom"


def test_tool_call_event_duration():
    """make_tool_call_event stores duration_ms in metrics."""
    from paperbot.application.collaboration.agent_events import make_tool_call_event

    envelope = make_tool_call_event(
        tool_name="paper_search",
        arguments={},
        result_summary="ok",
        duration_ms=42.5,
        run_id="run-3",
        trace_id="trace-3",
    )

    assert "duration_ms" in envelope.metrics
    assert envelope.metrics["duration_ms"] == 42.5


def test_audit_uses_constants():
    """_audit.py references EventType.TOOL_ERROR and EventType.TOOL_RESULT instead of raw strings."""
    audit_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "src"
        / "paperbot"
        / "mcp"
        / "tools"
        / "_audit.py"
    )
    source = audit_path.read_text()

    assert "EventType.TOOL_ERROR" in source, (
        "_audit.py must use EventType.TOOL_ERROR instead of raw 'error' string"
    )
    assert "EventType.TOOL_RESULT" in source, (
        "_audit.py must use EventType.TOOL_RESULT instead of raw 'tool_result' string"
    )
    # Confirm the raw string literals are no longer used for the type= argument
    assert 'type="error"' not in source, (
        "_audit.py must not use raw type='error' — use EventType.TOOL_ERROR"
    )
    assert "type=\"tool_result\"" not in source, (
        "_audit.py must not use raw type='tool_result' — use EventType.TOOL_RESULT"
    )


def test_file_change_event_type():
    """EventType.FILE_CHANGE is the string 'file_change'."""
    from paperbot.application.collaboration.message_schema import EventType
    assert EventType.FILE_CHANGE == "file_change"
    assert isinstance(EventType.FILE_CHANGE, str)


# --- Codex delegation constants (Phase 10 / CDX-03) ---

def test_codex_dispatched_event_type():
    """EventType.CODEX_DISPATCHED is the string 'codex_dispatched'."""
    from paperbot.application.collaboration.message_schema import EventType
    assert EventType.CODEX_DISPATCHED == "codex_dispatched"
    assert isinstance(EventType.CODEX_DISPATCHED, str)


def test_codex_accepted_event_type():
    """EventType.CODEX_ACCEPTED is the string 'codex_accepted'."""
    from paperbot.application.collaboration.message_schema import EventType
    assert EventType.CODEX_ACCEPTED == "codex_accepted"
    assert isinstance(EventType.CODEX_ACCEPTED, str)


def test_codex_completed_event_type():
    """EventType.CODEX_COMPLETED is the string 'codex_completed'."""
    from paperbot.application.collaboration.message_schema import EventType
    assert EventType.CODEX_COMPLETED == "codex_completed"
    assert isinstance(EventType.CODEX_COMPLETED, str)


def test_codex_failed_event_type():
    """EventType.CODEX_FAILED is the string 'codex_failed'."""
    from paperbot.application.collaboration.message_schema import EventType
    assert EventType.CODEX_FAILED == "codex_failed"
    assert isinstance(EventType.CODEX_FAILED, str)
