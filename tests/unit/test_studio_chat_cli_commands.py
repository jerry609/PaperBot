from __future__ import annotations

import asyncio
from pathlib import Path

from paperbot.api.routes import studio_chat
from paperbot.api.routes.studio_chat import (
    ChatMessage,
    StudioChatRequest,
    StudioCommandRequest,
    build_claude_cli_command_args,
    build_prompt_with_context,
    build_management_command,
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
