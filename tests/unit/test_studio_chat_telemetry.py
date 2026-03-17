from __future__ import annotations

from paperbot.api.routes.studio_chat import (
    _StudioTelemetryState,
    _build_cli_telemetry_events,
    _parse_cli_event,
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


def test_build_cli_telemetry_events_non_delegation_tool_does_not_emit_worker_run():
    state = _make_state()

    events = _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_non_delegate",
                        "name": "Bash",
                        "input": {
                            "command": "echo ok",
                            "runtime": "codex",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=12.0,
    )

    assert [event.type for event in events] == [
        EventType.AGENT_WORKING,
        EventType.TOOL_CALL,
    ]


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
    assert dispatched[2].payload["runtime"] == "codex"
    assert dispatched[2].payload["interruptible"] is False
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


def test_build_cli_telemetry_events_opencode_delegation_completion_emits_terminal_events():
    state = _make_state()

    _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_opencode_finish",
                        "name": "spawn_agent",
                        "input": {
                            "runtime": "opencode",
                            "description": "Finish the OpenCode handoff",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=50.0,
    )

    completed = _build_cli_telemetry_events(
        {
            "type": "tool_result",
            "tool_name": "spawn_agent",
            "tool_use_id": "toolu_opencode_finish",
            "content": "delegation completed",
        },
        state,
        now_monotonic=51.0,
    )

    assert [event.type for event in completed] == [
        EventType.TOOL_RESULT,
        EventType.CODEX_COMPLETED,
        EventType.AGENT_COMPLETED,
    ]
    assert completed[1].payload["assignee"].startswith("opencode-")
    assert completed[2].agent_name.startswith("opencode-")


def test_build_cli_telemetry_events_claude_agent_flow_tracks_worker_activity():
    state = _make_state()

    dispatched = _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tooluse_agent",
                        "name": "Agent",
                        "input": {
                            "description": "Get current git branch",
                            "prompt": "Run `git -C /home/master1/PaperBot branch --show-current`.",
                            "subagent_type": "codex-worker",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=60.0,
    )

    assert [event.type for event in dispatched] == [
        EventType.AGENT_WORKING,
        EventType.TOOL_CALL,
        EventType.CODEX_DISPATCHED,
        EventType.AGENT_STARTED,
    ]
    codex_assignee = dispatched[2].payload["assignee"]
    assert codex_assignee.startswith("codex-")

    progress = _build_cli_telemetry_events(
        {
            "type": "system",
            "subtype": "task_progress",
            "tool_use_id": "tooluse_agent",
            "description": "Running Show current git branch",
            "last_tool_name": "Bash",
        },
        state,
        now_monotonic=60.5,
    )

    assert [event.type for event in progress] == [EventType.AGENT_WORKING]
    assert progress[0].agent_name == codex_assignee

    nested_tool = _build_cli_telemetry_events(
        {
            "type": "assistant",
            "parent_tool_use_id": "tooluse_agent",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tooluse_bash",
                        "name": "Bash",
                        "input": {
                            "command": "git -C /home/master1/PaperBot branch --show-current",
                            "description": "Show current git branch",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=61.0,
    )

    assert [event.type for event in nested_tool] == [
        EventType.AGENT_WORKING,
        EventType.TOOL_CALL,
    ]
    assert nested_tool[0].agent_name == codex_assignee
    assert nested_tool[1].agent_name == codex_assignee

    nested_result = _build_cli_telemetry_events(
        {
            "type": "user",
            "parent_tool_use_id": "tooluse_agent",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tooluse_bash",
                        "content": "test/milestone-v1.2",
                        "is_error": False,
                    }
                ]
            },
        },
        state,
        now_monotonic=61.5,
    )

    assert [event.type for event in nested_result] == [EventType.TOOL_RESULT]
    assert nested_result[0].agent_name == codex_assignee

    completed = _build_cli_telemetry_events(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tooluse_agent",
                        "content": [
                            {
                                "type": "text",
                                "text": "The current branch is `test/milestone-v1.2`.",
                            }
                        ],
                        "is_error": False,
                    }
                ]
            },
        },
        state,
        now_monotonic=62.0,
    )

    assert [event.type for event in completed] == [
        EventType.TOOL_RESULT,
        EventType.CODEX_COMPLETED,
        EventType.AGENT_COMPLETED,
    ]
    assert completed[1].payload["assignee"] == codex_assignee
    assert completed[2].agent_name == codex_assignee


def test_build_cli_telemetry_events_claude_worker_uses_worker_assignee_prefix():
    state = _make_state()

    dispatched = _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_claude_worker",
                        "name": "Agent",
                        "input": {
                            "description": "Review the proposed plan",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=70.0,
    )

    assert dispatched[2].type == EventType.CODEX_DISPATCHED
    assert dispatched[2].payload["runtime"] == "claude"
    assert dispatched[2].payload["assignee"].startswith("claude-worker-")


def test_parse_cli_event_system_init_emits_session_metadata():
    state = _make_state()

    events = _parse_cli_event(
        {
            "type": "system",
            "subtype": "init",
            "session_id": "5807135f-411e-4365-9b27-9e39ca13c3d5",
            "permissionMode": "default",
            "model": "claude-sonnet-4-6",
        },
        state,
    )

    assert len(events) == 1
    assert events[0].type == "progress"
    assert events[0].data["cli_event"] == "session_init"
    assert events[0].data["cli_session_id"] == "5807135f-411e-4365-9b27-9e39ca13c3d5"
    assert state.cli_session_id == "5807135f-411e-4365-9b27-9e39ca13c3d5"


def test_parse_cli_event_emits_approval_required_for_worker_result():
    state = _make_state()
    state.cli_session_id = "5807135f-411e-4365-9b27-9e39ca13c3d5"

    _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tooluse_agent",
                        "name": "Agent",
                        "input": {
                            "description": "Get current git branch",
                            "prompt": "Run `git -C /home/master1/PaperBot branch --show-current`.",
                            "subagent_type": "codex-worker",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=60.0,
    )

    events = _parse_cli_event(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tooluse_agent",
                        "content": [
                            {
                                "type": "text",
                                "text": "The command requires approval from you to run. Could you approve the `git -C /home/master1/PaperBot branch --show-current` command, or run it yourself and share the output?",
                            },
                            {
                                "type": "text",
                                "text": "agentId: afec8e10340629da4 (for resuming to continue this agent's work if needed)",
                            },
                        ],
                        "is_error": False,
                    }
                ]
            },
        },
        state,
    )

    assert len(events) == 2
    assert events[0].data["cli_event"] == "tool_result"
    assert events[1].data["cli_event"] == "approval_required"
    assert events[1].data["command"] == "git -C /home/master1/PaperBot branch --show-current"
    assert events[1].data["worker_agent_id"] == "afec8e10340629da4"
    assert events[1].data["cli_session_id"] == "5807135f-411e-4365-9b27-9e39ca13c3d5"


def test_parse_cli_event_emits_bridge_result_for_structured_worker_output():
    state = _make_state()

    _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tooluse_agent",
                        "name": "Agent",
                        "input": {
                            "description": "Implement telemetry bridge",
                            "subagent_type": "codex-worker",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=70.0,
    )

    events = _parse_cli_event(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tooluse_agent",
                        "content": [
                            {
                                "type": "text",
                                "text": """```json
{
  "version": "1",
  "executor": "codex",
  "task_kind": "code",
  "status": "completed",
  "summary": "Implemented the telemetry bridge.",
  "artifacts": [
    { "kind": "file", "label": "studio_chat.py", "path": "src/paperbot/api/routes/studio_chat.py" }
  ],
  "payload": {
    "files_changed": ["src/paperbot/api/routes/studio_chat.py"],
    "checks": [{ "label": "pytest", "status": "passed" }]
  }
}
```""",
                            }
                        ],
                        "is_error": False,
                    }
                ]
            },
        },
        state,
    )

    assert len(events) == 2
    assert events[0].data["cli_event"] == "tool_result"
    assert events[1].data["cli_event"] == "bridge_result"
    assert events[1].data["bridge_result"]["task_kind"] == "code"
    assert events[1].data["bridge_result"]["status"] == "completed"
    assert events[1].data["bridge_result"]["summary"] == "Implemented the telemetry bridge."
    assert events[1].data["bridge_result"]["delegation"]["task_id"] == "tooluse_agent"
    assert events[1].data["bridge_result"]["delegation"]["worker_run_id"] == "worker-run-tooluse_agen"
    assert events[1].data["bridge_result"]["payload"]["delegation_task_id"] == "tooluse_agent"


def test_parse_cli_event_emits_structured_approval_request_from_bridge_result():
    state = _make_state()
    state.cli_session_id = "5807135f-411e-4365-9b27-9e39ca13c3d5"

    _build_cli_telemetry_events(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tooluse_agent",
                        "name": "Agent",
                        "input": {
                            "description": "Inspect branch",
                            "subagent_type": "codex-worker",
                        },
                    }
                ]
            },
        },
        state,
        now_monotonic=80.0,
    )

    events = _parse_cli_event(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tooluse_agent",
                        "content": [
                            {
                                "type": "text",
                                "text": """{
  "version": "1",
  "executor": "codex",
  "task_kind": "approval_required",
  "status": "approval_required",
  "summary": "Need approval to run a read-only git command.",
  "artifacts": [],
  "payload": {
    "command": "git -C /home/master1/PaperBot branch --show-current",
    "resume_hint": { "worker_agent_id": "afec8e10340629da4" }
  }
}""",
                            }
                        ],
                        "is_error": False,
                    }
                ]
            },
        },
        state,
    )

    assert len(events) == 3
    assert events[1].data["cli_event"] == "bridge_result"
    assert events[2].data["cli_event"] == "approval_required"
    assert events[2].data["message"] == "Need approval to run a read-only git command."
    assert events[2].data["command"] == "git -C /home/master1/PaperBot branch --show-current"
    assert events[2].data["worker_agent_id"] == "afec8e10340629da4"
    assert events[2].data["bridge_result"]["delegation"]["task_id"] == "tooluse_agent"
    assert events[2].data["bridge_result"]["delegation"]["worker_run_id"] == "worker-run-tooluse_agen"
