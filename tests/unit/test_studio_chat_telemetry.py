from __future__ import annotations

from paperbot.api.routes.studio_chat import (
    _StudioTelemetryState,
    _build_cli_telemetry_events,
)
from paperbot.application.collaboration.message_schema import EventType


def _make_state() -> _StudioTelemetryState:
    return _StudioTelemetryState(
        run_id="run-studio-1",
        trace_id="trace-studio-1",
        session_id="studio-session-1",
        stage="plan",
    )


def test_build_cli_telemetry_events_tool_use_emits_working_and_tool_call():
    state = _make_state()

    events = _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "src/demo.py"},
                    }
                ]
            },
        },
        state,
        now_monotonic=10.0,
    )

    assert [event.type for event in events] == [
        EventType.AGENT_WORKING,
        EventType.TOOL_CALL,
    ]
    assert events[1].payload["tool"] == "read_file"
    assert events[1].payload["arguments"] == {"path": "src/demo.py"}


def test_build_cli_telemetry_events_write_result_emits_file_change():
    state = _make_state()

    _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_write",
                        "name": "create_file",
                        "input": {"path": "src/generated.py", "content": "print('ok')"},
                    }
                ]
            },
        },
        state,
        now_monotonic=5.0,
    )

    events = _build_cli_telemetry_events(
        {
            "type": "tool_result",
            "tool_name": "create_file",
            "tool_use_id": "toolu_write",
            "content": "created file",
        },
        state,
        now_monotonic=5.25,
    )

    assert [event.type for event in events] == [
        EventType.TOOL_RESULT,
        EventType.FILE_CHANGE,
    ]
    assert events[1].payload["path"] == "src/generated.py"
    assert events[1].payload["status"] == "created"


def test_build_cli_telemetry_events_codex_delegation_emits_dispatch_and_completion():
    state = _make_state()

    dispatched = _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_delegate",
                        "name": "task",
                        "input": {
                            "assignee": "codex",
                            "prompt": "Implement telemetry bridge",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=20.0,
    )

    assert [event.type for event in dispatched] == [
        EventType.AGENT_WORKING,
        EventType.TOOL_CALL,
        EventType.CODEX_DISPATCHED,
        EventType.AGENT_STARTED,
    ]
    assert dispatched[2].payload["assignee"].startswith("codex-")
    assert dispatched[2].payload["task_title"] == "Implement telemetry bridge"
    assert dispatched[3].agent_name.startswith("codex-")

    completed = _build_cli_telemetry_events(
        {
            "type": "tool_result",
            "tool_name": "task",
            "tool_use_id": "toolu_delegate",
            "content": "delegated task finished",
        },
        state,
        now_monotonic=21.5,
    )

    assert [event.type for event in completed] == [
        EventType.TOOL_RESULT,
        EventType.CODEX_COMPLETED,
        EventType.AGENT_COMPLETED,
    ]
    assert completed[1].payload["assignee"].startswith("codex-")


def test_build_cli_telemetry_events_delegation_error_emits_failed_events():
    state = _make_state()

    _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_fail",
                        "name": "delegate",
                        "input": {
                            "backend": "codex",
                            "instructions": "Refactor the parser",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=30.0,
    )

    failed = _build_cli_telemetry_events(
        {
            "type": "tool_result",
            "tool_name": "delegate",
            "tool_use_id": "toolu_fail",
            "is_error": True,
            "error": "subagent crashed",
            "content": "subagent crashed",
        },
        state,
        now_monotonic=31.0,
    )

    assert [event.type for event in failed] == [
        EventType.TOOL_ERROR,
        EventType.CODEX_FAILED,
        EventType.AGENT_ERROR,
    ]
    assert failed[1].payload["reason_code"] == "tool_error"
    assert failed[1].payload["error"] == "subagent crashed"


def test_build_cli_telemetry_events_opencode_delegation_uses_opencode_assignee():
    state = _make_state()

    dispatched = _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_opencode",
                        "name": "spawn_agent",
                        "input": {
                            "runtime": "opencode",
                            "description": "Write the dashboard shell",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=40.0,
    )

    assert dispatched[2].type == EventType.CODEX_DISPATCHED
    assert dispatched[2].payload["assignee"].startswith("opencode-")
    assert dispatched[3].agent_name.startswith("opencode-")
