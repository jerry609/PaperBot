from __future__ import annotations

import asyncio
from pathlib import Path

from paperbot.api.routes import studio_chat
from paperbot.api.routes.studio_chat import (
    ChatMessage,
    StudioChatRequest,
    StudioCommandRequest,
    UploadedFileAttachment,
    _StudioTelemetryState,
    _persist_uploaded_files,
    build_claude_cli_command_args,
    build_prompt_with_context,
    build_management_command,
    _make_studio_mode_changed_event,
    _make_studio_session_init_event,
)


def test_build_claude_cli_command_args_includes_print_mode_flags_and_effective_mode():
    request = StudioChatRequest(
        message="Implement the telemetry bridge",
        mode="Code",
        model="claude-sonnet-4-6",
        continue_last=True,
        resume_session="resume-123",
        cli_session_id="123e4567-e89b-12d3-a456-426614174000",
        agent="reviewer",
        mcp_config=["./mcp.json", '{"server":"local"}'],
        tools=["Bash", "Read"],
        allowed_tools=["Bash(git:*)", "Read"],
        add_dirs=["../shared", "./workspace"],
        settings='{"theme":"slate"}',
        effort="high",
    )

    args = build_claude_cli_command_args(
        request,
        effective_mode="Plan",
        prompt="prompt-body",
    )

    assert args == [
        "--model",
        "claude-sonnet-4-6",
        "--continue",
        "--resume",
        "resume-123",
        "--session-id",
        "123e4567-e89b-12d3-a456-426614174000",
        "--agent",
        "reviewer",
        "--add-dir",
        "../shared",
        "./workspace",
        "--mcp-config",
        "./mcp.json",
        '{"server":"local"}',
        "--tools",
        "Bash,Read",
        "--allowed-tools",
        "Bash(git:*),Read",
        "--settings",
        '{"theme":"slate"}',
        "--effort",
        "high",
        "--permission-mode",
        "plan",
        "-p",
        "prompt-body",
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def test_build_claude_cli_command_args_enables_full_access_for_code_mode(monkeypatch):
    monkeypatch.setenv("PAPERBOT_STUDIO_ENABLE_CODE_MODE", "true")
    request = StudioChatRequest(
        message="Refactor the monitor layout",
        mode="Code",
        model="sonnet",
        permission_profile="full_access",
    )

    args = build_claude_cli_command_args(
        request,
        effective_mode="Code",
        prompt="prompt-body",
    )

    assert args == [
        "--model",
        "sonnet",
        "--allow-dangerously-skip-permissions",
        "--permission-mode",
        "bypassPermissions",
        "-p",
        "prompt-body",
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def test_persist_uploaded_files_writes_temp_files(tmp_path, monkeypatch):
    upload_dir = tmp_path / "studio-upload"
    upload_dir.mkdir()
    monkeypatch.setattr(studio_chat.tempfile, "mkdtemp", lambda prefix: str(upload_dir))

    prompt_entries, add_dirs = _persist_uploaded_files(
        [
            UploadedFileAttachment(
                id="upload-1",
                name="../report final.pdf",
                type="application/pdf",
                size=11,
                data="aGVsbG8gd29ybGQ=",
            ),
        ],
        session_id="session-123",
    )

    assert add_dirs == [str(upload_dir.resolve())]
    assert len(prompt_entries) == 1
    saved_path = Path(prompt_entries[0].split(" -> ", 1)[1])
    assert prompt_entries[0].startswith("../report final.pdf -> ")
    assert saved_path.read_bytes() == b"hello world"
    assert saved_path.name == "01-report_final.pdf"


def test_build_prompt_with_context_includes_recent_history_blocks():
    prompt = build_prompt_with_context(
        "Apply the same fix to the API client",
        paper=None,
        mode="Code",
        history=[
            ChatMessage(role="user", content="Inspect the failing loader test"),
            ChatMessage(role="assistant", content="The test is failing on a missing fixture."),
        ],
    )

    assert "# Conversation History" in prompt
    assert "## User\nInspect the failing loader test" in prompt
    assert "## Assistant\nThe test is failing on a missing fixture." in prompt
    assert "# User Request\nApply the same fix to the API client" in prompt


def test_build_prompt_with_context_includes_attached_workspace_files():
    prompt = build_prompt_with_context(
        "",
        paper=None,
        mode="Code",
        attached_files=["src/app.ts", "docs/spec.md", "src/app.ts"],
        uploaded_files=["report.pdf -> /tmp/upload/report.pdf", "diagram.png -> /tmp/upload/diagram.png"],
    )

    assert "# Attached Workspace Files" in prompt
    assert "- src/app.ts" in prompt
    assert "- docs/spec.md" in prompt
    assert prompt.count("- src/app.ts") == 1
    assert "# Uploaded Files" in prompt
    assert "report.pdf -> /tmp/upload/report.pdf" in prompt
    assert "Please inspect the attached file(s)." in prompt


def test_build_management_command_uses_safe_noninteractive_defaults(monkeypatch):
    monkeypatch.setattr(studio_chat, "find_claude_cli", lambda: "/usr/local/bin/claude")
    monkeypatch.setattr(studio_chat, "find_opencode_cli", lambda: "/usr/local/bin/opencode")

    assert build_management_command(
        StudioCommandRequest(runtime="claude", command="mcp"),
    ) == ["/usr/local/bin/claude", "mcp", "list"]
    assert build_management_command(
        StudioCommandRequest(runtime="claude", command="auth"),
    ) == ["/usr/local/bin/claude", "auth", "status"]
    assert build_management_command(
        StudioCommandRequest(runtime="opencode", command="agent"),
    ) == ["/usr/local/bin/opencode", "agent", "list"]
    assert build_management_command(
        StudioCommandRequest(runtime="opencode", command="providers"),
    ) == ["/usr/local/bin/opencode", "providers", "list"]


def test_build_management_command_rejects_unsupported_subcommand(monkeypatch):
    monkeypatch.setattr(studio_chat, "find_claude_cli", lambda: "/usr/local/bin/claude")

    request = StudioCommandRequest(runtime="claude", command="chat")

    try:
        build_management_command(request)
    except ValueError as exc:
        assert "Unsupported claude command" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported Claude command")


def test_make_studio_session_init_event_exposes_managed_session_metadata():
    request = StudioChatRequest(
        message="Map the managed chat contract",
        mode="Code",
        model="sonnet",
    )
    state = _StudioTelemetryState(
        run_id="run-studio-1",
        trace_id="trace-studio-1",
        session_id="studio-session-1",
        stage="plan",
    )

    event = _make_studio_session_init_event(
        state,
        request=request,
        effective_mode="Plan",
        transport="claude_cli_print",
        cwd="/tmp/project",
    )

    assert event.type == "status"
    assert event.event == "status"
    assert event.data["subtype"] == "init"
    assert event.data["session_id"] == "studio-session-1"
    assert event.data["chat_surface"] == "managed_session"
    assert event.data["chat_transport"] == "claude_cli_print"
    assert event.data["preferred_chat_transport"] == "claude_agent_sdk"
    assert event.data["mode"] == "Plan"
    assert event.data["requested_mode"] == "Code"
    assert event.data["permission_profile"] == "default"
    assert event.data["permission_mode"] == "plan"
    assert event.data["cwd"] == "/tmp/project"
    assert "doctor" in event.data["slash_commands"]
    assert "full_access" in event.data["permission_profiles"]
    assert "mcp" in event.data["runtime_commands"]


def test_make_studio_mode_changed_event_only_emits_when_mode_changes():
    request = StudioChatRequest(message="Ship it", mode="Code", model="sonnet")

    changed = _make_studio_mode_changed_event(
        request=request,
        effective_mode="Plan",
    )
    assert changed is not None
    assert changed.data["subtype"] == "mode_changed"
    assert changed.data["mode"] == "Plan"
    assert changed.data["requested_mode"] == "Code"

    unchanged = _make_studio_mode_changed_event(
        request=request,
        effective_mode="Code",
    )
    assert unchanged is None


def test_studio_command_route_executes_safe_management_command(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(studio_chat, "find_claude_cli", lambda: "/usr/local/bin/claude")

    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, env, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout

        class Result:
            returncode = 0
            stdout = "configured-mcp\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(studio_chat.subprocess, "run", fake_run)

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(studio_chat.asyncio, "to_thread", fake_to_thread)

    payload = asyncio.run(
        studio_chat.studio_command(
            StudioCommandRequest(
                runtime="claude",
                command="mcp",
                project_dir=str(tmp_path),
            )
        )
    )
    assert payload["ok"] is True
    assert payload["stdout"] == "configured-mcp\n"
    assert payload["command"] == ["/usr/local/bin/claude", "mcp", "list"]
    assert payload["cwd"] == str(tmp_path)
    assert captured["cmd"] == ["/usr/local/bin/claude", "mcp", "list"]
    assert captured["cwd"] == str(tmp_path)
