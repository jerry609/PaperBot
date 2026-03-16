"""
Studio chat transport for the PaperBot Studio shell.

Chat turns run Claude CLI in print mode and stream structured NDJSON events.
Standalone utility commands such as ``claude mcp`` and ``claude doctor`` go
through a separate management-command path instead of the chat stream.
"""

import asyncio
import base64
import importlib.util
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, AsyncGenerator

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from paperbot.application.collaboration.agent_events import make_lifecycle_event
from paperbot.application.collaboration.message_schema import EventType, make_event, new_run_id, new_trace_id

from ..streaming import StreamEvent, sse_response

router = APIRouter()

Mode = Literal["Code", "Plan", "Ask"]
Effort = Literal["low", "medium", "high", "max"]
PermissionProfile = Literal["default", "full_access"]
StudioRuntimeName = Literal["claude", "opencode"]
StudioChatTransport = Literal["claude_agent_sdk", "claude_cli_print", "anthropic_api"]
DEFAULT_STUDIO_MODEL = "sonnet"
_LEGACY_MODEL_ALIASES = {
    "claude-sonnet-4-5": "sonnet",
    "claude-opus-4-5": "opus",
    "claude-haiku-4-5": "haiku",
}
_API_FALLBACK_MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-5-20250514",
    "opus": "claude-opus-4-5-20250514",
    "haiku": "claude-haiku-4-5-20250514",
}
_KNOWN_CLAUDE_MODEL_ALIASES = ["sonnet", "opus"]
_ALLOWED_MANAGEMENT_COMMANDS: Dict[str, set[str]] = {
    "claude": {"agents", "mcp", "auth", "doctor"},
    "opencode": {"agent", "mcp", "providers", "models"},
}
_CLAUDE_MODEL_SETTING_ENV_KEYS = (
    "PAPERBOT_STUDIO_DEFAULT_MODEL",
    "CLAUDE_CODE_MODEL",
    "ANTHROPIC_MODEL",
)


@dataclass(frozen=True)
class StudioDefaultModelDetection:
    model: str
    source: str


class PaperContext(BaseModel):
    title: str
    abstract: str
    method_section: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class UploadedFileAttachment(BaseModel):
    id: str
    name: str
    type: str = "application/octet-stream"
    size: int = 0
    data: str


class StudioChatRequest(BaseModel):
    message: str
    mode: Mode = "Code"
    model: str = Field(default_factory=lambda: detect_claude_default_model())
    permission_profile: PermissionProfile = "default"
    paper: Optional[PaperContext] = None
    project_dir: Optional[str] = None
    history: List[ChatMessage] = []
    attached_files: List[str] = []
    uploaded_files: List[UploadedFileAttachment] = []
    session_id: Optional[str] = None
    context_pack_id: Optional[str] = None
    continue_last: bool = False
    resume_session: Optional[str] = None
    cli_session_id: Optional[str] = None
    agent: Optional[str] = None
    mcp_config: List[str] = []
    tools: List[str] = []
    allowed_tools: List[str] = []
    add_dirs: List[str] = []
    settings: Optional[str] = None
    effort: Optional[Effort] = None


class StudioCommandRequest(BaseModel):
    runtime: StudioRuntimeName = "claude"
    command: str
    args: str = ""
    project_dir: Optional[str] = None
    timeout_ms: int = 15000


def find_claude_cli() -> Optional[str]:
    """Find Claude CLI executable path."""
    nvm_candidates = sorted(
        (Path.home() / ".nvm" / "versions" / "node").glob("v*/bin/claude"),
        reverse=True,
    )

    # Check common locations
    candidates = [
        shutil.which("claude"),
        *[str(path) for path in nvm_candidates],
        os.path.expanduser("~/.npm-global/bin/claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/opt/homebrew/bin/claude",
        "/usr/local/bin/claude",
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def find_opencode_cli() -> Optional[str]:
    nvm_candidates = sorted(
        (Path.home() / ".nvm" / "versions" / "node").glob("v*/bin/opencode"),
        reverse=True,
    )

    candidates = [
        shutil.which("opencode"),
        *[str(path) for path in nvm_candidates],
        os.path.expanduser("~/.npm-global/bin/opencode"),
        os.path.expanduser("~/.local/bin/opencode"),
        "/opt/homebrew/bin/opencode",
        "/usr/local/bin/opencode",
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def is_code_mode_enabled() -> bool:
    raw = os.getenv("PAPERBOT_STUDIO_ENABLE_CODE_MODE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def has_claude_agent_sdk() -> bool:
    """Return True when the Python Claude Agent SDK is importable."""
    return importlib.util.find_spec("claude_agent_sdk") is not None


def current_studio_chat_transport(*, claude_path: Optional[str]) -> StudioChatTransport:
    """Return the transport currently used by Studio chat."""
    if claude_path:
        return "claude_cli_print"
    return "anthropic_api"


def preferred_studio_chat_transport() -> StudioChatTransport:
    """Return the long-term transport we align Studio chat with."""
    return "claude_agent_sdk"


def _studio_supported_slash_commands() -> List[str]:
    return ["help", "status", "new", "clear", "plan", "model", "agents", "mcp", "auth", "doctor"]


def _studio_supported_permission_profiles() -> List[str]:
    return ["default", "full_access"]


def _make_studio_session_init_event(
    state: "_StudioTelemetryState",
    *,
    request: "StudioChatRequest",
    effective_mode: Mode,
    transport: StudioChatTransport,
    cwd: Optional[str] = None,
) -> StreamEvent:
    resolved_model = get_model_id(
        request.model,
        for_cli=True,
        project_dir=request.project_dir,
    )
    data: Dict[str, Any] = {
        "subtype": "init",
        "session_id": state.session_id,
        "chat_surface": "managed_session",
        "chat_transport": transport,
        "preferred_chat_transport": preferred_studio_chat_transport(),
        "claude_agent_sdk_available": has_claude_agent_sdk(),
        "mode": effective_mode,
        "requested_mode": request.mode,
        "permission_profile": request.permission_profile,
        "permission_mode": resolve_permission_mode(
            request.mode,
            request.permission_profile,
        ),
        "model": resolved_model,
        "known_model_aliases": _KNOWN_CLAUDE_MODEL_ALIASES,
        "slash_commands": _studio_supported_slash_commands(),
        "permission_profiles": _studio_supported_permission_profiles(),
        "runtime_commands": sorted(_ALLOWED_MANAGEMENT_COMMANDS["claude"]),
    }
    if cwd:
        data["cwd"] = cwd
    return StreamEvent(type="status", event="status", data=data)


def _make_studio_mode_changed_event(
    *,
    request: "StudioChatRequest",
    effective_mode: Mode,
) -> Optional[StreamEvent]:
    if effective_mode == request.mode:
        return None
    return StreamEvent(
        type="status",
        event="status",
        data={
            "subtype": "mode_changed",
            "mode": effective_mode,
            "requested_mode": request.mode,
            "reason": f"Requested {request.mode} mode is unavailable; using {effective_mode} instead.",
        },
    )


def resolve_execution_mode(mode: Mode) -> Mode:
    if mode == "Code" and not is_code_mode_enabled():
        return "Plan"
    return mode


def resolve_permission_mode(
    mode: Mode,
    permission_profile: PermissionProfile = "default",
) -> str:
    """Map Studio mode and permission profile to Claude CLI permission mode."""
    effective_mode = resolve_execution_mode(mode)
    if effective_mode == "Code":
        return "bypassPermissions" if permission_profile == "full_access" else "acceptEdits"
    if effective_mode == "Plan":
        return "plan"
    return "default"


def get_mode_flags(
    mode: Mode,
    permission_profile: PermissionProfile = "default",
) -> List[str]:
    """Map mode to Claude CLI permission flags."""
    permission_mode = resolve_permission_mode(mode, permission_profile)
    if permission_mode == "bypassPermissions":
        return [
            "--allow-dangerously-skip-permissions",
            "--permission-mode",
            "bypassPermissions",
        ]
    if permission_mode == "acceptEdits":
        return ["--permission-mode", "acceptEdits"]
    if permission_mode == "plan":
        return ["--permission-mode", "plan"]
    return []


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_requested_model(
    model: Optional[str],
    *,
    project_dir: Optional[str] = None,
) -> str:
    requested = (model or "").strip()
    if not requested:
        return detect_claude_default_model(project_dir)
    return _LEGACY_MODEL_ALIASES.get(requested, requested)


def get_model_id(
    model: Optional[str],
    for_cli: bool = False,
    *,
    project_dir: Optional[str] = None,
) -> str:
    """Resolve a requested model for Claude CLI or API fallback.

    Claude Code accepts either short aliases such as ``sonnet`` / ``opus`` or
    a full model name. The Studio UI therefore forwards the user-provided value
    instead of forcing an outdated hard-coded list.
    """
    normalized = _normalize_requested_model(model, project_dir=project_dir)
    if for_cli:
        return normalized
    return _API_FALLBACK_MODEL_ALIASES.get(normalized, normalized)


def _find_nearest_claude_settings_file(start_dir: Path, filename: str) -> Optional[Path]:
    search_root = start_dir.resolve()
    for directory in (search_root, *search_root.parents):
        candidate = directory / ".claude" / filename
        if candidate.is_file():
            return candidate
    return None


def _read_model_from_claude_settings(path: Path) -> Optional[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    model = _clean_optional_text(payload.get("model")) if isinstance(payload.get("model"), str) else None
    if not model:
        return None

    return _LEGACY_MODEL_ALIASES.get(model, model)


def detect_claude_default_model_details(project_dir: Optional[str] = None) -> StudioDefaultModelDetection:
    for env_key in _CLAUDE_MODEL_SETTING_ENV_KEYS:
        env_value = _clean_optional_text(os.getenv(env_key))
        if env_value:
            return StudioDefaultModelDetection(
                model=_LEGACY_MODEL_ALIASES.get(env_value, env_value),
                source=f"env:{env_key}",
            )

    workspace_dir = Path(project_dir).resolve() if project_dir else Path(os.getcwd()).resolve()

    local_settings = _find_nearest_claude_settings_file(workspace_dir, "settings.local.json")
    if local_settings is not None:
        model = _read_model_from_claude_settings(local_settings)
        if model:
            return StudioDefaultModelDetection(model=model, source="workspace-local")

    project_settings = _find_nearest_claude_settings_file(workspace_dir, "settings.json")
    if project_settings is not None:
        model = _read_model_from_claude_settings(project_settings)
        if model:
            return StudioDefaultModelDetection(model=model, source="workspace")

    user_settings = Path.home() / ".claude" / "settings.json"
    model = _read_model_from_claude_settings(user_settings)
    if model:
        return StudioDefaultModelDetection(model=model, source="user")

    return StudioDefaultModelDetection(model=DEFAULT_STUDIO_MODEL, source="fallback")


def detect_claude_default_model(project_dir: Optional[str] = None) -> str:
    return detect_claude_default_model_details(project_dir).model


def _normalize_text_items(values: List[str]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        cleaned = _clean_optional_text(value)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _append_multi_value_flag(cmd: List[str], flag: str, values: List[str]) -> None:
    normalized = _normalize_text_items(values)
    if normalized:
        cmd.extend([flag, *normalized])


def _append_joined_flag(cmd: List[str], flag: str, values: List[str]) -> None:
    normalized = _normalize_text_items(values)
    if normalized:
        cmd.extend([flag, ",".join(normalized)])


def _merge_text_items(*groups: List[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in _normalize_text_items(group):
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _normalize_attached_files(values: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_optional_text(value)
        if not cleaned or "\x00" in cleaned:
            continue
        key = cleaned.replace("\\", "/")
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def build_user_request_content(
    message: str,
    attached_files: Optional[List[str]] = None,
    uploaded_files: Optional[List[str]] = None,
) -> str:
    """Build the effective user request text, including selected workspace files."""
    normalized_message = _clean_optional_text(message) or "Please inspect the attached file(s)."
    normalized_files = _normalize_attached_files(attached_files or [])
    normalized_uploaded_files = _normalize_text_items(uploaded_files or [])
    if not normalized_files and not normalized_uploaded_files:
        return normalized_message

    sections: List[str] = [normalized_message]

    if normalized_files:
        attachment_lines = "\n".join(f"- {path}" for path in normalized_files)
        sections.extend(
            [
                "# Attached Workspace Files",
                attachment_lines,
                "Treat these paths as user-selected workspace context. Read or edit them directly when needed.",
            ]
        )

    if normalized_uploaded_files:
        uploaded_lines = "\n".join(f"- {path}" for path in normalized_uploaded_files)
        sections.extend(
            [
                "# Uploaded Files",
                uploaded_lines,
                "These files were uploaded from the Studio UI for this turn. Open them directly when they are relevant.",
            ]
        )

    return "\n\n".join(sections)


def _sanitize_uploaded_filename(filename: str) -> str:
    name = os.path.basename(filename.strip()) or "upload.bin"
    safe = "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in name)
    safe = safe.strip("._") or "upload.bin"
    return safe[:120]


def _persist_uploaded_files(
    uploads: List[UploadedFileAttachment],
    *,
    session_id: str,
) -> tuple[List[str], List[str]]:
    if not uploads:
        return [], []

    upload_dir = Path(tempfile.mkdtemp(prefix=f"paperbot-studio-upload-{session_id[:8]}-")).resolve()
    prompt_entries: List[str] = []

    for index, upload in enumerate(uploads, start=1):
        payload = upload.data.strip()
        if not payload:
            raise ValueError(f"Uploaded file '{upload.name}' is empty")
        if "," in payload and "base64" in payload[:64]:
            payload = payload.split(",", 1)[1]

        try:
            content = base64.b64decode(payload, validate=False)
        except Exception as exc:
            raise ValueError(f"Uploaded file '{upload.name}' could not be decoded") from exc

        safe_name = _sanitize_uploaded_filename(upload.name)
        file_path = upload_dir / f"{index:02d}-{safe_name}"
        file_path.write_bytes(content)
        prompt_entries.append(f"{upload.name} -> {file_path}")

    return prompt_entries, [str(upload_dir)]


def build_claude_cli_command_args(
    request: StudioChatRequest,
    *,
    effective_mode: Mode,
    prompt: str,
    extra_add_dirs: Optional[List[str]] = None,
) -> List[str]:
    # Studio chat uses print mode rather than an interactive Claude TTY.
    model_id = get_model_id(
        request.model,
        for_cli=True,
        project_dir=request.project_dir,
    )
    cmd: List[str] = ["--model", model_id]

    if request.continue_last:
        cmd.append("--continue")

    resume_session = _clean_optional_text(request.resume_session)
    if resume_session:
        cmd.extend(["--resume", resume_session])

    cli_session_id = _clean_optional_text(request.cli_session_id)
    if cli_session_id:
        cmd.extend(["--session-id", cli_session_id])

    agent = _clean_optional_text(request.agent)
    if agent:
        cmd.extend(["--agent", agent])

    _append_multi_value_flag(cmd, "--add-dir", _merge_text_items(request.add_dirs, extra_add_dirs or []))
    _append_multi_value_flag(cmd, "--mcp-config", request.mcp_config)
    _append_joined_flag(cmd, "--tools", request.tools)
    _append_joined_flag(cmd, "--allowed-tools", request.allowed_tools)

    settings = _clean_optional_text(request.settings)
    if settings:
        cmd.extend(["--settings", settings])

    if request.effort:
        cmd.extend(["--effort", request.effort])

    cmd.extend(get_mode_flags(effective_mode, request.permission_profile))
    cmd.extend(["-p", prompt, "--output-format", "stream-json", "--verbose"])
    return cmd


def build_management_command(
    request: StudioCommandRequest,
) -> List[str]:
    # Runtime utility commands are intentionally limited to a small allowlist.
    runtime = request.runtime
    command = request.command.strip()
    if command not in _ALLOWED_MANAGEMENT_COMMANDS[runtime]:
        allowed = ", ".join(sorted(_ALLOWED_MANAGEMENT_COMMANDS[runtime]))
        raise ValueError(f"Unsupported {runtime} command '{command}'. Allowed: {allowed}")

    binary = find_claude_cli() if runtime == "claude" else find_opencode_cli()
    if not binary:
        raise ValueError(f"{runtime} CLI not found")

    cmd = [binary, command]
    extra_args = shlex.split(request.args) if request.args.strip() else []
    if not extra_args:
        if runtime == "claude" and command == "mcp":
            extra_args = ["list"]
        elif runtime == "claude" and command == "auth":
            extra_args = ["status"]
        elif runtime == "opencode" and command in {"agent", "mcp", "providers"}:
            extra_args = ["list"]
    cmd.extend(extra_args)
    return cmd


def _format_history_for_prompt(history: List[ChatMessage]) -> str:
    lines: List[str] = []

    for msg in history[-10:]:
        content = msg.content.strip()
        if not content:
            continue
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"## {role}\n{content}")

    if not lines:
        return ""

    return "# Conversation History\n\n" + "\n\n".join(lines)


def build_prompt_with_context(
    message: str,
    paper: Optional[PaperContext],
    mode: Mode,
    history: Optional[List[ChatMessage]] = None,
    attached_files: Optional[List[str]] = None,
    uploaded_files: Optional[List[str]] = None,
) -> str:
    """Build the prompt with paper context if available."""
    parts = []

    if paper:
        parts.append(f"# Paper Context\n**Title:** {paper.title}\n\n**Abstract:** {paper.abstract}")
        if paper.method_section:
            parts.append(f"\n**Method Section:** {paper.method_section}")
        parts.append("\n---\n")

    history_block = _format_history_for_prompt(history or [])
    if history_block:
        parts.append(history_block)
        parts.append("\n---\n")

    if mode == "Code":
        parts.append("You are helping implement this research paper as working code. ")
    elif mode == "Plan":
        parts.append("You are creating an implementation plan for this research paper. Do not write code, only plan. ")
    else:
        parts.append("You are answering questions about this research paper. ")

    parts.append(f"\n# User Request\n{build_user_request_content(message, attached_files, uploaded_files)}")

    return "\n".join(parts)


log = logging.getLogger(__name__)


@dataclass
class _ToolInvocation:
    tool_name: str
    tool_id: str
    arguments: Dict[str, Any]
    started_at: float
    agent_name: str = "claude"
    role: str = "orchestrator"


@dataclass
class _PendingDelegation:
    tool_name: str
    tool_id: str
    assignee: str
    task_id: str
    task_title: str
    runtime: str
    worker_run_id: str
    control_mode: str = "mirrored"
    interruptible: bool = False


@dataclass
class _StudioTelemetryState:
    run_id: str
    trace_id: str
    session_id: str
    stage: str
    tool_invocations: List[_ToolInvocation] = field(default_factory=list)
    pending_delegations: List[_PendingDelegation] = field(default_factory=list)
    runtime_terminal_emitted: bool = False


def _append_eventlog(event_log, envelope) -> None:
    if event_log is None or envelope is None:
        return
    try:
        event_log.append(envelope)
    except Exception:
        log.debug("Failed to append studio telemetry event", exc_info=True)


def _make_studio_lifecycle_event(
    state: _StudioTelemetryState,
    *,
    status: str,
    agent_name: str,
    stage: str,
    detail: Optional[str] = None,
    role: str = "orchestrator",
):
    return make_lifecycle_event(
        status=status,
        agent_name=agent_name,
        run_id=state.run_id,
        trace_id=state.trace_id,
        workflow="studio_chat",
        stage=stage,
        role=role,
        detail=detail,
    )


def _make_studio_tool_event(
    state: _StudioTelemetryState,
    *,
    event_type: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    result_summary: str = "",
    error: Optional[str] = None,
    duration_ms: float = 0.0,
    agent_name: str = "claude",
    role: str = "orchestrator",
):
    return make_event(
        run_id=state.run_id,
        trace_id=state.trace_id,
        workflow="studio_chat",
        stage="tool_call",
        attempt=0,
        agent_name=agent_name,
        role=role,
        type=event_type,
        payload={
            "tool": tool_name,
            "arguments": arguments or {},
            "result_summary": result_summary,
            "error": error,
        },
        metrics={"duration_ms": duration_ms},
    )


def _make_file_change_event(
    state: _StudioTelemetryState,
    *,
    path: str,
    status: str,
    agent_name: str = "claude",
    role: str = "orchestrator",
):
    return make_event(
        run_id=state.run_id,
        trace_id=state.trace_id,
        workflow="studio_chat",
        stage="tool_call",
        attempt=0,
        agent_name=agent_name,
        role=role,
        type=EventType.FILE_CHANGE,
        payload={
            "path": path,
            "status": status,
        },
    )


def _make_delegation_event(
    state: _StudioTelemetryState,
    *,
    event_type: str,
    delegation: _PendingDelegation,
    error: Optional[str] = None,
    reason_code: Optional[str] = None,
):
    payload: Dict[str, Any] = {
        "task_id": delegation.task_id,
        "task_title": delegation.task_title,
        "session_id": state.session_id,
        "assignee": delegation.assignee,
        "runtime": delegation.runtime,
        "worker_run_id": delegation.worker_run_id,
        "control_mode": delegation.control_mode,
        "interruptible": delegation.interruptible,
    }
    if error is not None:
        payload["error"] = error
    if reason_code is not None:
        payload["reason_code"] = reason_code

    return make_event(
        run_id=state.run_id,
        trace_id=state.trace_id,
        workflow="studio_chat",
        stage="delegation",
        attempt=0,
        agent_name=delegation.assignee,
        role="worker",
        type=event_type,
        payload=payload,
    )


def _drain_pending_delegation_failures(
    state: _StudioTelemetryState,
    *,
    error: str,
    reason_code: str,
) -> List:
    emitted: List = []
    while state.pending_delegations:
        delegation = state.pending_delegations.pop(0)
        emitted.append(
            _make_delegation_event(
                state,
                event_type=EventType.CODEX_FAILED,
                delegation=delegation,
                error=error,
                reason_code=reason_code,
            )
        )
        emitted.append(
            _make_studio_lifecycle_event(
                state,
                status=EventType.AGENT_ERROR,
                agent_name=delegation.assignee,
                stage="delegation",
                detail=error,
            )
        )
    return emitted


def _extract_tool_path(tool_input: Dict[str, Any]) -> Optional[str]:
    for key in ("path", "file_path", "filename", "target_file", "target_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_like_file_write(tool_name: str) -> bool:
    normalized = tool_name.strip().lower()
    return normalized in {
        "write",
        "write_file",
        "edit",
        "multiedit",
        "str_replace_editor",
        "create_file",
        "replace",
    }


def _tool_key(tool_name: str, tool_id: str) -> str:
    return f"{tool_name}:{tool_id}" if tool_id else tool_name


def _find_tool_invocation(
    state: _StudioTelemetryState,
    *,
    tool_name: str = "",
    tool_id: str = "",
) -> Optional[_ToolInvocation]:
    if tool_id:
        for invocation in reversed(state.tool_invocations):
            if invocation.tool_id == tool_id:
                return invocation
    if tool_name:
        for invocation in reversed(state.tool_invocations):
            if invocation.tool_name == tool_name:
                return invocation
    return None


def _pop_tool_invocation(
    state: _StudioTelemetryState,
    *,
    tool_name: str,
    tool_id: str,
) -> Optional[_ToolInvocation]:
    if tool_id:
        for idx, invocation in enumerate(state.tool_invocations):
            if invocation.tool_id == tool_id:
                return state.tool_invocations.pop(idx)

    for idx, invocation in enumerate(state.tool_invocations):
        if invocation.tool_name == tool_name:
            return state.tool_invocations.pop(idx)
    return None


def _truncate_text_field(value: Any, max_len: int = 120) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value or "")
    return _truncate(text.strip(), max_len)


def _extract_task_title(tool_input: Dict[str, Any], fallback: str) -> str:
    for key in ("task_title", "title", "summary", "description", "prompt", "message", "instructions"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value.strip(), 96)
    return fallback


def _is_delegation_tool(tool_name: str, tool_input: Dict[str, Any]) -> bool:
    normalized_name = tool_name.strip().lower()
    if normalized_name in {
        "agent",
        "task",
        "spawn_agent",
        "delegate",
        "delegate_task",
        "dispatch_agent",
        "subagent",
        "teamcreate",
    }:
        return True

    for key in (
        "subagent_type",
        "delegate_to",
        "delegate_to_runtime",
        "teammate_name",
        "team_name",
    ):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return True

    return False


def _infer_subagent_runtime(tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
    normalized_name = tool_name.strip().lower()
    if not _is_delegation_tool(tool_name, tool_input):
        return None

    candidate_values: List[str] = [normalized_name]
    for key in (
        "agent",
        "assignee",
        "delegate_to",
        "runtime",
        "executor",
        "runner",
        "subagent",
        "subagent_type",
        "backend",
        "target",
        "teammate_name",
        "team_name",
    ):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            candidate_values.append(value.strip().lower())

    for value in candidate_values:
        if "opencode" in value or "open code" in value:
            return "opencode"
        if "codex" in value:
            return "codex"
        if "team" in value or "teammate" in value:
            return "claude"
        if value in {"claude", "cc"} or value.startswith("claude-") or value.startswith("cc-"):
            return "claude"

    if normalized_name in {"agent", "teamcreate"}:
        return "claude"
    return "worker"


def _register_delegation(
    state: _StudioTelemetryState,
    *,
    tool_name: str,
    tool_id: str,
    tool_input: Dict[str, Any],
) -> Optional[_PendingDelegation]:
    runtime = _infer_subagent_runtime(tool_name, tool_input)
    if runtime is None:
        return None

    suffix = (tool_id or f"{len(state.pending_delegations) + 1}").replace(":", "-")
    assignee_prefix = {
        "claude": "claude-worker",
        "codex": "codex",
        "opencode": "opencode",
        "worker": "worker",
    }.get(runtime, runtime)
    assignee = f"{assignee_prefix}-{suffix[:6]}"
    task_id = tool_id or f"studio-delegation-{len(state.pending_delegations) + 1}"
    worker_run_id = f"worker-run-{suffix[:12]}"
    fallback_title = f"{tool_name} delegation"
    task_title = _extract_task_title(tool_input, fallback_title)

    delegation = _PendingDelegation(
        tool_name=tool_name,
        tool_id=tool_id,
        assignee=assignee,
        task_id=task_id,
        task_title=task_title,
        runtime=runtime,
        worker_run_id=worker_run_id,
    )
    state.pending_delegations.append(delegation)
    return delegation


def _pop_delegation(
    state: _StudioTelemetryState,
    *,
    tool_name: str,
    tool_id: str,
) -> Optional[_PendingDelegation]:
    if tool_id:
        key = _tool_key(tool_name, tool_id)
        for idx, delegation in enumerate(state.pending_delegations):
            if _tool_key(delegation.tool_name, delegation.tool_id) == key:
                return state.pending_delegations.pop(idx)

    for idx, delegation in enumerate(state.pending_delegations):
        if delegation.tool_name == tool_name:
            return state.pending_delegations.pop(idx)
    return None


def _find_pending_delegation_by_tool_id(
    state: _StudioTelemetryState,
    *,
    tool_id: str,
) -> Optional[_PendingDelegation]:
    if not tool_id:
        return None
    for delegation in reversed(state.pending_delegations):
        if delegation.tool_id == tool_id:
            return delegation
    return None


def _stringify_tool_result_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        if text_parts:
            return "\n".join(text_parts)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _emit_tool_result_events(
    state: _StudioTelemetryState,
    *,
    tool_name: str,
    tool_id: str,
    content: Any,
    is_error: bool,
    error: Optional[str],
    now: float,
) -> List:
    emitted: List = []
    invocation = _pop_tool_invocation(state, tool_name=tool_name, tool_id=tool_id)
    resolved_tool_name = invocation.tool_name if invocation is not None else (tool_name or "unknown")
    duration_ms = max(0.0, (now - invocation.started_at) * 1000) if invocation else 0.0
    arguments = invocation.arguments if invocation is not None else {}
    result_content = _stringify_tool_result_content(content)
    result_summary = _truncate_text_field(result_content, 240)
    error_text = _truncate_text_field(error or result_content or "", 240) if is_error else None
    owner_agent_name = invocation.agent_name if invocation is not None else "claude"
    owner_role = invocation.role if invocation is not None else "orchestrator"

    emitted.append(
        _make_studio_tool_event(
            state,
            event_type=EventType.TOOL_ERROR if is_error else EventType.TOOL_RESULT,
            tool_name=resolved_tool_name,
            arguments=arguments,
            result_summary=result_summary,
            error=error_text,
            duration_ms=duration_ms,
            agent_name=owner_agent_name,
            role=owner_role,
        )
    )

    if invocation is not None and not is_error and _looks_like_file_write(invocation.tool_name):
        path = _extract_tool_path(invocation.arguments)
        if path:
            emitted.append(
                _make_file_change_event(
                    state,
                    path=path,
                    status="created" if "create" in invocation.tool_name.lower() else "modified",
                    agent_name=owner_agent_name,
                    role=owner_role,
                )
            )

    delegation = _pop_delegation(state, tool_name=resolved_tool_name, tool_id=tool_id)
    if delegation is not None:
        emitted.append(
            _make_delegation_event(
                state,
                event_type=EventType.CODEX_FAILED if is_error else EventType.CODEX_COMPLETED,
                delegation=delegation,
                error=error_text,
                reason_code="tool_error" if is_error else None,
            )
        )
        emitted.append(
            _make_studio_lifecycle_event(
                state,
                status=EventType.AGENT_ERROR if is_error else EventType.AGENT_COMPLETED,
                agent_name=delegation.assignee,
                stage="delegation",
                detail=error_text if is_error else delegation.task_title,
            )
        )

    return emitted


def _is_tool_result_error(line_data: Dict[str, Any]) -> bool:
    if bool(line_data.get("is_error")):
        return True
    if line_data.get("subtype") == "error":
        return True
    error_value = line_data.get("error")
    return isinstance(error_value, str) and bool(error_value.strip())


def _build_cli_telemetry_events(
    line_data: Dict[str, Any],
    state: _StudioTelemetryState,
    *,
    now_monotonic: Optional[float] = None,
) -> List:
    now = time.monotonic() if now_monotonic is None else now_monotonic
    etype = str(line_data.get("type", "")).strip()
    emitted: List = []

    if etype == "assistant":
        parent_tool_use_id = str(line_data.get("parent_tool_use_id") or "").strip()
        parent_delegation = _find_pending_delegation_by_tool_id(
            state,
            tool_id=parent_tool_use_id,
        ) if parent_tool_use_id else None
        msg = line_data.get("message", {})
        content_blocks = msg.get("content", []) if isinstance(msg, dict) else []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", "")).strip()
            if btype == "thinking":
                thinking = str(block.get("thinking", "")).strip()
                if thinking:
                    emitted.append(
                        _make_studio_lifecycle_event(
                            state,
                            status=EventType.AGENT_WORKING,
                            agent_name="claude",
                            stage=state.stage,
                            detail=_truncate(thinking, 160),
                        )
                    )
            elif btype == "tool_use":
                tool_name = str(block.get("name", "unknown")).strip() or "unknown"
                tool_id = str(block.get("id", "")).strip()
                tool_input = block.get("input", {})
                arguments = tool_input if isinstance(tool_input, dict) else {}
                owner_agent_name = parent_delegation.assignee if parent_delegation is not None else "claude"
                owner_role = "worker" if parent_delegation is not None else "orchestrator"

                state.tool_invocations.append(
                    _ToolInvocation(
                        tool_name=tool_name,
                        tool_id=tool_id,
                        arguments=arguments,
                        started_at=now,
                        agent_name=owner_agent_name,
                        role=owner_role,
                    )
                )

                emitted.append(
                    _make_studio_lifecycle_event(
                        state,
                        status=EventType.AGENT_WORKING,
                        agent_name=owner_agent_name,
                        stage="delegation" if parent_delegation is not None else state.stage,
                        detail=f"Using {tool_name}",
                        role=owner_role,
                    )
                )
                emitted.append(
                    _make_studio_tool_event(
                        state,
                        event_type=EventType.TOOL_CALL,
                        tool_name=tool_name,
                        arguments=arguments,
                        result_summary="started",
                        agent_name=owner_agent_name,
                        role=owner_role,
                    )
                )

                delegation = _register_delegation(
                    state,
                    tool_name=tool_name,
                    tool_id=tool_id,
                    tool_input=arguments,
                )
                if delegation is not None:
                    emitted.append(
                        _make_delegation_event(
                            state,
                            event_type=EventType.CODEX_DISPATCHED,
                            delegation=delegation,
                        )
                    )
                    emitted.append(
                        _make_studio_lifecycle_event(
                            state,
                            status=EventType.AGENT_STARTED,
                            agent_name=delegation.assignee,
                            stage="delegation",
                            detail=delegation.task_title,
                        )
                    )

    elif etype == "tool_result":
        tool_name = str(line_data.get("tool_name", "")).strip() or "unknown"
        tool_id = str(line_data.get("tool_use_id") or line_data.get("tool_id") or "").strip()
        is_error = _is_tool_result_error(line_data)
        emitted.extend(
            _emit_tool_result_events(
                state,
                tool_name=tool_name,
                tool_id=tool_id,
                content=line_data.get("content", ""),
                is_error=is_error,
                error=str(line_data.get("error") or "").strip() or None,
                now=now,
            )
        )

    elif etype == "user":
        msg = line_data.get("message", {})
        content_blocks = msg.get("content", []) if isinstance(msg, dict) else []
        for block in content_blocks:
            if not isinstance(block, dict) or str(block.get("type", "")).strip() != "tool_result":
                continue
            tool_id = str(block.get("tool_use_id") or "").strip()
            emitted.extend(
                _emit_tool_result_events(
                    state,
                    tool_name="",
                    tool_id=tool_id,
                    content=block.get("content", ""),
                    is_error=bool(block.get("is_error")),
                    error=None,
                    now=now,
                )
            )

    elif etype == "system":
        subtype = str(line_data.get("subtype", "")).strip()
        tool_id = str(line_data.get("tool_use_id") or "").strip()
        delegation = _find_pending_delegation_by_tool_id(state, tool_id=tool_id)
        if delegation is not None and subtype == "task_progress":
            detail = str(line_data.get("description") or "").strip()
            last_tool_name = str(line_data.get("last_tool_name") or "").strip()
            if last_tool_name:
                detail = f"{detail} ({last_tool_name})" if detail else f"Using {last_tool_name}"
            if detail:
                emitted.append(
                    _make_studio_lifecycle_event(
                        state,
                        status=EventType.AGENT_WORKING,
                        agent_name=delegation.assignee,
                        stage="delegation",
                        detail=_truncate(detail, 160),
                        role="worker",
                    )
                )

    elif etype == "result":
        is_error = line_data.get("subtype") == "error" or _is_tool_result_error(line_data)
        detail = _truncate_text_field(
            line_data.get("error") or line_data.get("result") or "Studio chat turn completed",
            240,
        )
        state.runtime_terminal_emitted = True
        emitted.append(
            _make_studio_lifecycle_event(
                state,
                status=EventType.AGENT_ERROR if is_error else EventType.AGENT_COMPLETED,
                agent_name="claude",
                stage=state.stage,
                detail=detail,
            )
        )
        if is_error:
            emitted.extend(
                _drain_pending_delegation_failures(
                    state,
                    error=detail,
                    reason_code="runtime_error",
                )
            )

    return emitted


def _load_runtime_allowed_dirs() -> List[Path]:
    f = Path("data/runbook_allowed_dirs.json")
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    dirs: List[Path] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            try:
                dirs.append(Path(item).resolve())
            except Exception:
                continue
    return dirs


def _allowed_workdir_prefixes() -> List[Path]:
    prefixes: List[Path] = [Path(tempfile.gettempdir()).resolve()]
    try:
        prefixes.append(Path.cwd().resolve())
    except Exception:
        pass
    try:
        home_dir = Path.home().resolve()
        prefixes.append((home_dir / "Documents").resolve(strict=False))
    except Exception:
        pass

    extra = os.getenv("PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES", "").strip()
    if extra:
        for p in extra.split(","):
            p = p.strip()
            if p:
                try:
                    prefixes.append(Path(p).expanduser().resolve())
                except Exception:
                    continue

    prefixes.extend(_load_runtime_allowed_dirs())

    unique: List[Path] = []
    seen: set[str] = set()
    for p in prefixes:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _runtime_allowlist_mutation_enabled() -> bool:
    return os.getenv("PAPERBOT_RUNBOOK_ALLOWLIST_MUTATION", "false").lower() == "true"


def _is_under_prefix(path: Path, prefix: Path) -> bool:
    path_real = os.path.realpath(str(path))
    prefix_real = os.path.realpath(str(prefix))
    return path_real == prefix_real or path_real.startswith(prefix_real + os.sep)


def _path_is_existing_dir(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir()
    except Exception:
        return False


def _preferred_studio_workspace_dir(actual_cwd: Path) -> Path:
    allowed_prefixes = _allowed_workdir_prefixes()
    temp_root = Path(tempfile.gettempdir()).resolve()
    repo_like_cwd = "paperbot" in str(actual_cwd).lower()

    if not repo_like_cwd and _path_is_existing_dir(actual_cwd):
        return actual_cwd

    for prefix in allowed_prefixes:
        if prefix in {temp_root, actual_cwd}:
            continue
        if _path_is_existing_dir(prefix):
            return prefix

    if _path_is_existing_dir(actual_cwd):
        return actual_cwd

    for prefix in allowed_prefixes:
        if _path_is_existing_dir(prefix):
            return prefix

    return temp_root


def _resolve_cli_project_dir(raw: Optional[str]) -> Path:
    """Resolve and validate project_dir used by studio CLI execution."""
    if not raw:
        return Path.cwd().resolve()

    cleaned = raw.strip()
    if not cleaned or "\x00" in cleaned:
        raise ValueError("invalid project_dir")

    if cleaned == "~":
        normalized = str(Path.home())
    elif cleaned.startswith("~/"):
        normalized = str(Path.home() / cleaned[2:])
    else:
        normalized = cleaned

    # Normalize to real path and reconstruct from an allowed prefix.
    # This avoids resolving an arbitrary user-controlled path directly.
    if not os.path.isabs(normalized):
        normalized = str((Path.cwd() / normalized).resolve(strict=False))
    normalized_real = os.path.realpath(normalized)

    resolved: Optional[Path] = None
    for prefix in _allowed_workdir_prefixes():
        prefix_real = os.path.realpath(str(prefix))
        if normalized_real == prefix_real:
            resolved = prefix
            break
        if normalized_real.startswith(prefix_real + os.sep):
            suffix = normalized_real[len(prefix_real):].lstrip("/\\")
            candidate = (prefix / suffix).resolve(strict=False) if suffix else prefix
            if _is_under_prefix(candidate, prefix):
                resolved = candidate
                break
            raise ValueError("project_dir is not allowed")

    if resolved is None:
        raise ValueError("project_dir is not allowed")

    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("project_dir must be an existing directory")
    return resolved


def _load_context_pack(pack_id: str) -> Optional[Dict[str, Any]]:
    """Load a context pack from the database by ID."""
    try:
        from ...infrastructure.stores.repro_context_store import SqlAlchemyReproContextStore

        store = SqlAlchemyReproContextStore()
        return store.get(pack_id)
    except Exception as exc:
        log.warning("Failed to load context pack %s: %s", pack_id, exc)
        return None


def _format_context_pack_markdown(pack: Dict[str, Any]) -> str:
    """Format a context pack as a Markdown document for Claude CLI to read."""
    lines: list[str] = []

    lines.append("# Reproduction Context Pack")
    lines.append("")

    # Paper metadata
    paper = pack.get("paper", {})
    if paper:
        lines.append("## Paper")
        if paper.get("title"):
            lines.append(f"**Title:** {paper['title']}")
        if paper.get("authors"):
            authors = paper["authors"]
            if isinstance(authors, list):
                lines.append(f"**Authors:** {', '.join(authors)}")
            else:
                lines.append(f"**Authors:** {authors}")
        if paper.get("year"):
            lines.append(f"**Year:** {paper['year']}")
        if paper.get("arxiv_id"):
            lines.append(f"**arXiv:** {paper['arxiv_id']}")
        if paper.get("doi"):
            lines.append(f"**DOI:** {paper['doi']}")
        lines.append("")

    # Objective
    if pack.get("objective"):
        lines.append("## Objective")
        lines.append(pack["objective"])
        lines.append("")

    # Task roadmap
    roadmap = pack.get("task_roadmap", [])
    if roadmap:
        lines.append("## Task Roadmap")
        lines.append("")
        for i, step in enumerate(roadmap, 1):
            title = step.get("title", f"Step {i}")
            lines.append(f"### Step {i}: {title}")
            if step.get("description"):
                lines.append(step["description"])
            if step.get("acceptance_criteria"):
                lines.append("")
                lines.append("**Acceptance criteria:**")
                criteria = step["acceptance_criteria"]
                if isinstance(criteria, list):
                    for c in criteria:
                        lines.append(f"- {c}")
                else:
                    lines.append(f"- {criteria}")
            lines.append("")

    # Observations
    observations = pack.get("observations", [])
    if observations:
        lines.append("## Observations")
        lines.append("")
        for obs in observations:
            obs_type = obs.get("type", "note")
            title = obs.get("title", "Untitled")
            confidence = obs.get("confidence", 0)
            lines.append(f"### [{obs_type.upper()}] {title} (confidence: {confidence:.0%})")
            if obs.get("content"):
                lines.append(obs["content"])
            elif obs.get("description"):
                lines.append(obs["description"])
            if obs.get("code_snippet"):
                lines.append("")
                lang = obs.get("language", "")
                lines.append(f"```{lang}")
                lines.append(obs["code_snippet"])
                lines.append("```")
            lines.append("")

    # Warnings
    warnings = pack.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def _ensure_context_pack_on_disk(pack_id: str, project_dir: str) -> Optional[str]:
    """Load context pack from DB and write CONTEXT.md into project_dir.

    Returns the file path on success, None on failure.
    """
    pack = _load_context_pack(pack_id)
    if pack is None:
        return None

    md = _format_context_pack_markdown(pack)
    target = Path(project_dir) / "CONTEXT.md"
    try:
        target.write_text(md, encoding="utf-8")
        return str(target)
    except OSError as exc:
        log.warning("Failed to write CONTEXT.md to %s: %s", project_dir, exc)
        return None


def _parse_cli_content_blocks(content_blocks: list) -> list[StreamEvent]:
    """Convert Claude CLI assistant content blocks into StreamEvents."""
    events: list[StreamEvent] = []
    for block in content_blocks:
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if text:
                events.append(StreamEvent(
                    type="progress",
                    data={"cli_event": "text", "text": text},
                ))

        elif btype == "tool_use":
            events.append(StreamEvent(
                type="progress",
                data={
                    "cli_event": "tool_use",
                    "tool_name": block.get("name", "unknown"),
                    "tool_input": block.get("input", {}),
                    "tool_id": block.get("id", ""),
                },
            ))

        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                events.append(StreamEvent(
                    type="progress",
                    data={"cli_event": "thinking", "text": thinking},
                ))

    return events


def _parse_user_tool_result_blocks(
    line_data: Dict[str, Any],
    telemetry_state: Optional[_StudioTelemetryState] = None,
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    msg = line_data.get("message", {})
    content_blocks = msg.get("content", []) if isinstance(msg, dict) else []

    for block in content_blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue

        tool_id = str(block.get("tool_use_id", "")).strip()
        invocation = (
            _find_tool_invocation(telemetry_state, tool_id=tool_id)
            if telemetry_state is not None and tool_id
            else None
        )
        events.append(StreamEvent(
            type="progress",
            data={
                "cli_event": "tool_result",
                "tool_name": invocation.tool_name if invocation is not None else "tool",
                "tool_id": tool_id,
                "is_error": bool(block.get("is_error")),
                "content": _truncate(_stringify_tool_result_content(block.get("content", "")), 2000),
            },
        ))

    return events


def _parse_cli_event(
    line_data: Dict[str, Any],
    telemetry_state: Optional[_StudioTelemetryState] = None,
) -> list[StreamEvent]:
    """Parse a single NDJSON line from `claude -p --output-format stream-json`.

    Claude CLI stream-json emits one JSON object per line:
    - {"type":"assistant","message":{...}} — assistant turn with content blocks
    - {"type":"tool_result","tool_name":"...","content":"..."} — tool output
    - {"type":"user","message":{"content":[{"type":"tool_result",...}]}} — current tool output shape
    - {"type":"result","subtype":"success","result":"...","cost_usd":...} — final
    - {"type":"system",...} — session init (ignored)
    """
    etype = line_data.get("type", "")
    events: list[StreamEvent] = []

    if etype == "assistant":
        msg = line_data.get("message", {})
        content_blocks = msg.get("content", [])
        events.extend(_parse_cli_content_blocks(content_blocks))

    elif etype == "tool_result":
        events.append(StreamEvent(
            type="progress",
            data={
                "cli_event": "tool_result",
                "tool_name": line_data.get("tool_name", ""),
                "tool_id": line_data.get("tool_use_id") or line_data.get("tool_id") or "",
                "content": _truncate(_stringify_tool_result_content(line_data.get("content", "")), 2000),
            },
        ))

    elif etype == "user":
        events.extend(_parse_user_tool_result_blocks(line_data, telemetry_state))

    elif etype == "result":
        events.append(StreamEvent(
            type="result",
            data={
                "cli_event": "done",
                "result": line_data.get("result", ""),
                "cost_usd": line_data.get("cost_usd"),
                "duration_ms": line_data.get("duration_ms"),
                "num_turns": line_data.get("num_turns"),
            },
        ))

    # Ignore "system" and other meta events
    return events


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len] + "..."


async def stream_claude_cli(
    request: StudioChatRequest,
    *,
    telemetry_state: _StudioTelemetryState,
    event_log=None,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream Claude CLI output as structured SSE events.

    Uses ``--output-format stream-json`` so we get real-time NDJSON events
    (text, tool_use, tool_result) instead of buffered plain text.
    """

    claude_path = find_claude_cli()

    if not claude_path:
        _append_eventlog(
            event_log,
            _make_studio_lifecycle_event(
                telemetry_state,
                status=EventType.AGENT_ERROR,
                agent_name="claude",
                stage=telemetry_state.stage,
                detail="Claude CLI not found",
                role="orchestrator",
            ),
        )
        telemetry_state.runtime_terminal_emitted = True
        yield StreamEvent(
            type="error",
            message="Claude CLI not found. Please install it with: npm install -g @anthropic-ai/claude-code"
        )
        return

    effective_mode = resolve_execution_mode(request.mode)
    transport = current_studio_chat_transport(claude_path=claude_path)
    resolved_model = get_model_id(
        request.model,
        for_cli=True,
        project_dir=request.project_dir,
    )

    _append_eventlog(
        event_log,
        _make_studio_lifecycle_event(
            telemetry_state,
            status=EventType.AGENT_STARTED,
            agent_name="claude",
            stage=telemetry_state.stage,
            detail=f"{effective_mode} turn started",
            role="orchestrator",
        ),
    )
    _append_eventlog(
        event_log,
        _make_studio_lifecycle_event(
            telemetry_state,
            status=EventType.AGENT_WORKING,
            agent_name="claude",
            stage=telemetry_state.stage,
            detail="Connecting to Claude CLI",
            role="orchestrator",
        ),
    )

    yield StreamEvent(
        type="progress",
        data={
            "phase": "Starting",
            "message": f"[{effective_mode}] Connecting to Claude CLI...",
            "model": resolved_model,
            "mode": effective_mode,
            "requested_mode": request.mode,
        }
    )

    try:
        try:
            cwd = str(_resolve_cli_project_dir(request.project_dir))
        except ValueError as exc:
            _append_eventlog(
                event_log,
                _make_studio_lifecycle_event(
                    telemetry_state,
                    status=EventType.AGENT_ERROR,
                    agent_name="claude",
                    stage=telemetry_state.stage,
                    detail=str(exc),
                    role="orchestrator",
                ),
            )
            telemetry_state.runtime_terminal_emitted = True
            yield StreamEvent(type="error", message=str(exc))
            return

        try:
            uploaded_file_entries, upload_add_dirs = await asyncio.to_thread(
                _persist_uploaded_files,
                request.uploaded_files,
                session_id=telemetry_state.session_id,
            )
        except ValueError as exc:
            _append_eventlog(
                event_log,
                _make_studio_lifecycle_event(
                    telemetry_state,
                    status=EventType.AGENT_ERROR,
                    agent_name="claude",
                    stage=telemetry_state.stage,
                    detail=str(exc),
                    role="orchestrator",
                ),
            )
            telemetry_state.runtime_terminal_emitted = True
            yield StreamEvent(type="error", message=str(exc))
            return

        prompt = build_prompt_with_context(
            request.message,
            request.paper,
            effective_mode,
            request.history,
            request.attached_files,
            uploaded_file_entries,
        )
        cmd = [
            claude_path,
            *build_claude_cli_command_args(
                request,
                effective_mode=effective_mode,
                prompt=prompt,
                extra_add_dirs=upload_add_dirs,
            ),
        ]

        yield _make_studio_session_init_event(
            telemetry_state,
            request=request,
            effective_mode=effective_mode,
            transport=transport,
            cwd=cwd,
        )
        mode_changed_event = _make_studio_mode_changed_event(
            request=request,
            effective_mode=effective_mode,
        )
        if mode_changed_event is not None:
            yield mode_changed_event

        # Write context pack to working directory so Claude CLI can read it
        if request.context_pack_id:
            pack_path = await asyncio.to_thread(
                _ensure_context_pack_on_disk, request.context_pack_id, cwd,
            )
            if pack_path:
                log.info("Wrote context pack to %s", pack_path)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env={**os.environ, "FORCE_COLOR": "0"},
        )
        stderr_chunks: list[str] = []

        async def _drain_stderr() -> None:
            if process.stderr is None:
                return
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    return
                stderr_chunks.append(chunk.decode("utf-8", errors="replace"))

        stderr_task = asyncio.create_task(_drain_stderr())

        _KEEPALIVE_SECONDS = 15
        line_buffer = ""

        # Read stdout line-by-line (NDJSON) with keepalive heartbeats.
        while True:
            try:
                chunk = await asyncio.wait_for(
                    process.stdout.read(4096),
                    timeout=_KEEPALIVE_SECONDS,
                )
            except asyncio.TimeoutError:
                yield StreamEvent(
                    type="progress",
                    data={
                        "keepalive": True,
                        "mode": effective_mode,
                        "requested_mode": request.mode,
                    },
                )
                continue

            if not chunk:
                break

            line_buffer += chunk.decode("utf-8", errors="replace")

            # Process complete lines (each is a JSON object)
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON CLI line: %s", line[:120])
                    continue

                for event in _parse_cli_event(data, telemetry_state):
                    yield event
                for envelope in _build_cli_telemetry_events(data, telemetry_state):
                    _append_eventlog(event_log, envelope)

        # Process any trailing data in buffer
        if line_buffer.strip():
            try:
                data = json.loads(line_buffer.strip())
                for event in _parse_cli_event(data, telemetry_state):
                    yield event
                for envelope in _build_cli_telemetry_events(data, telemetry_state):
                    _append_eventlog(event_log, envelope)
            except json.JSONDecodeError:
                pass

        await process.wait()
        await stderr_task

        if process.returncode != 0:
            error_msg = "".join(stderr_chunks).strip()
            if error_msg:
                if not telemetry_state.runtime_terminal_emitted:
                    _append_eventlog(
                        event_log,
                        _make_studio_lifecycle_event(
                            telemetry_state,
                            status=EventType.AGENT_ERROR,
                            agent_name="claude",
                            stage=telemetry_state.stage,
                            detail=error_msg,
                            role="orchestrator",
                        ),
                    )
                    for envelope in _drain_pending_delegation_failures(
                        telemetry_state,
                        error=error_msg,
                        reason_code="runtime_error",
                    ):
                        _append_eventlog(event_log, envelope)
                    telemetry_state.runtime_terminal_emitted = True
                yield StreamEvent(type="error", message=error_msg)
                return
        elif not telemetry_state.runtime_terminal_emitted:
            _append_eventlog(
                event_log,
                _make_studio_lifecycle_event(
                    telemetry_state,
                    status=EventType.AGENT_COMPLETED,
                    agent_name="claude",
                    stage=telemetry_state.stage,
                    detail="Studio chat turn completed",
                    role="orchestrator",
                ),
            )
            telemetry_state.runtime_terminal_emitted = True

    except FileNotFoundError:
        detail = f"Claude CLI not found at: {claude_path}"
        if not telemetry_state.runtime_terminal_emitted:
            _append_eventlog(
                event_log,
                _make_studio_lifecycle_event(
                    telemetry_state,
                    status=EventType.AGENT_ERROR,
                    agent_name="claude",
                    stage=telemetry_state.stage,
                    detail=detail,
                    role="orchestrator",
                ),
            )
            telemetry_state.runtime_terminal_emitted = True
        yield StreamEvent(
            type="error",
            message=detail
        )
    except Exception as e:
        detail = f"Claude CLI error: {str(e)}"
        if not telemetry_state.runtime_terminal_emitted:
            _append_eventlog(
                event_log,
                _make_studio_lifecycle_event(
                    telemetry_state,
                    status=EventType.AGENT_ERROR,
                    agent_name="claude",
                    stage=telemetry_state.stage,
                    detail=detail,
                    role="orchestrator",
                ),
            )
            for envelope in _drain_pending_delegation_failures(
                telemetry_state,
                error=detail,
                reason_code="runtime_error",
            ):
                _append_eventlog(event_log, envelope)
            telemetry_state.runtime_terminal_emitted = True
        yield StreamEvent(
            type="error",
            message=detail
        )


async def stream_anthropic_api(
    request: StudioChatRequest,
    *,
    telemetry_state: _StudioTelemetryState,
    event_log=None,
) -> AsyncGenerator[StreamEvent, None]:
    """Fallback: Stream response using Anthropic API directly."""
    effective_mode = resolve_execution_mode(request.mode)
    transport = current_studio_chat_transport(claude_path=None)
    resolved_model = get_model_id(
        request.model,
        for_cli=False,
        project_dir=request.project_dir,
    )

    _append_eventlog(
        event_log,
        _make_studio_lifecycle_event(
            telemetry_state,
            status=EventType.AGENT_STARTED,
            agent_name="claude",
            stage=telemetry_state.stage,
            detail=f"{effective_mode} turn started",
            role="orchestrator",
        ),
    )
    _append_eventlog(
        event_log,
        _make_studio_lifecycle_event(
            telemetry_state,
            status=EventType.AGENT_WORKING,
            agent_name="claude",
            stage=telemetry_state.stage,
            detail="Using Anthropic API fallback",
            role="orchestrator",
        ),
    )

    yield _make_studio_session_init_event(
        telemetry_state,
        request=request,
        effective_mode=effective_mode,
        transport=transport,
        cwd=request.project_dir,
    )
    mode_changed_event = _make_studio_mode_changed_event(
        request=request,
        effective_mode=effective_mode,
    )
    if mode_changed_event is not None:
        yield mode_changed_event

    yield StreamEvent(
        type="progress",
        data={
            "phase": "Processing",
            "message": f"[{effective_mode}] Thinking...",
            "mode": effective_mode,
            "requested_mode": request.mode,
        },
    )

    try:
        from ...infrastructure.llm.providers.anthropic_provider import AnthropicProvider

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            _append_eventlog(
                event_log,
                _make_studio_lifecycle_event(
                    telemetry_state,
                    status=EventType.AGENT_ERROR,
                    agent_name="claude",
                    stage=telemetry_state.stage,
                    detail="ANTHROPIC_API_KEY not set",
                    role="orchestrator",
                ),
            )
            telemetry_state.runtime_terminal_emitted = True
            yield StreamEvent(type="error", message="ANTHROPIC_API_KEY not set")
            return

        provider = AnthropicProvider(
            api_key=api_key,
            model_name=resolved_model,
            max_tokens=8192,
        )

        # Build system prompt based on mode
        if effective_mode == "Code":
            system = "You are an expert AI coding assistant. Generate high-quality, well-documented code. Include type hints and docstrings. Wrap code in appropriate markdown code blocks."
        elif effective_mode == "Plan":
            system = "You are a research implementation architect. Create detailed implementation plans WITHOUT generating actual code. Focus on architecture, components, and step-by-step planning."
        else:
            system = "You are a helpful research assistant. Answer questions clearly and concisely."

        if request.paper:
            system += f"\n\n# Paper Context\n**Title:** {request.paper.title}\n**Abstract:** {request.paper.abstract}"
            if request.paper.method_section:
                system += f"\n**Method Section:** {request.paper.method_section}"

        messages = [{"role": "system", "content": system}]

        for msg in request.history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append(
            {
                "role": "user",
                "content": build_user_request_content(
                    request.message,
                    request.attached_files,
                    [upload.name for upload in request.uploaded_files],
                ),
            }
        )

        full_content = ""
        async for chunk in provider.stream(messages):
            if chunk.delta:
                full_content += chunk.delta
                yield StreamEvent(
                    type="progress",
                    data={
                        "delta": chunk.delta,
                        "content": full_content,
                        "mode": effective_mode,
                        "requested_mode": request.mode,
                    }
                )

        yield StreamEvent(
            type="result",
            data={
                "content": full_content,
                "mode": effective_mode,
                "requested_mode": request.mode,
                "model": resolved_model,
            }
        )
        _append_eventlog(
            event_log,
            _make_studio_lifecycle_event(
                telemetry_state,
                status=EventType.AGENT_COMPLETED,
                agent_name="claude",
                stage=telemetry_state.stage,
                detail="Studio chat turn completed",
                role="orchestrator",
            ),
        )
        telemetry_state.runtime_terminal_emitted = True

    except Exception as e:
        detail = f"API error: {str(e)}"
        if not telemetry_state.runtime_terminal_emitted:
            _append_eventlog(
                event_log,
                _make_studio_lifecycle_event(
                    telemetry_state,
                    status=EventType.AGENT_ERROR,
                    agent_name="claude",
                    stage=telemetry_state.stage,
                    detail=detail,
                    role="orchestrator",
                ),
            )
            telemetry_state.runtime_terminal_emitted = True
        yield StreamEvent(type="error", message=detail)


async def studio_chat_stream(
    request: StudioChatRequest,
    *,
    telemetry_state: _StudioTelemetryState,
    event_log=None,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream Studio chat from the current managed-chat transport.

    Current transport order:
    1. Claude CLI print mode
    2. Direct Anthropic API fallback

    Preferred long-term route remains the Claude Agent SDK, matching the
    CodePilot-style managed session architecture.
    """

    # Check if Claude CLI is available
    claude_path = find_claude_cli()

    if claude_path:
        # Use Claude CLI
        async for event in stream_claude_cli(
            request,
            telemetry_state=telemetry_state,
            event_log=event_log,
        ):
            yield event
    else:
        # Fallback to Anthropic API
        yield StreamEvent(
            type="progress",
            data={
                "phase": "Info",
                "message": "Claude CLI not found, using Anthropic API directly",
            }
        )
        async for event in stream_anthropic_api(
            request,
            telemetry_state=telemetry_state,
            event_log=event_log,
        ):
            yield event


@router.post("/studio/chat")
async def studio_chat(http_request: Request, request: StudioChatRequest):
    """
    Interactive chat for DeepStudio with Claude CLI integration.

    Modes:
    - Code: Generate/modify code only when PAPERBOT_STUDIO_ENABLE_CODE_MODE is enabled
    - Plan: Create implementation plans (no execution)
    - Ask: Answer questions (no tools)

    Returns Server-Sent Events with streaming text.
    """
    run_id = new_run_id()
    trace_id = new_trace_id()
    session_id = request.session_id or f"studio-{run_id[:8]}"
    telemetry_state = _StudioTelemetryState(
        run_id=run_id,
        trace_id=trace_id,
        session_id=session_id,
        stage=resolve_execution_mode(request.mode).lower(),
    )
    event_log = getattr(http_request.app.state, "event_log", None)

    return sse_response(
        studio_chat_stream(
            request,
            telemetry_state=telemetry_state,
            event_log=event_log,
        ),
        workflow="studio_chat",
        run_id=run_id,
        trace_id=trace_id,
    )


@router.post("/studio/command")
async def studio_command(request: StudioCommandRequest):
    """Run a non-chat Claude Code / OpenCode management command and return its output."""
    try:
        cmd = build_management_command(request)
        try:
            cwd = str(_resolve_cli_project_dir(request.project_dir))
        except ValueError as exc:
            return {
                "ok": False,
                "command": cmd,
                "returncode": 1,
                "stdout": "",
                "stderr": str(exc),
                "cwd": request.project_dir,
            }

        timeout_seconds = max(1.0, min(request.timeout_ms / 1000.0, 60.0))

        def _run():
            return subprocess.run(
                cmd,
                cwd=cwd,
                env={**os.environ, "FORCE_COLOR": "0"},
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

        result = await asyncio.to_thread(_run)
        return {
            "ok": result.returncode == 0,
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "cwd": cwd,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "command": [],
            "returncode": 124,
            "stdout": "",
            "stderr": "Command timed out",
            "cwd": request.project_dir,
        }
    except ValueError as exc:
        return {
            "ok": False,
            "command": [],
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
            "cwd": request.project_dir,
        }


@router.get("/studio/status")
async def studio_status():
    """Check if Claude CLI is available."""
    claude_path = find_claude_cli()
    agent_sdk_available = has_claude_agent_sdk()
    chat_transport = current_studio_chat_transport(claude_path=claude_path)
    preferred_transport = preferred_studio_chat_transport()
    detected_model = detect_claude_default_model_details()
    opencode_path = find_opencode_cli()
    opencode_version = None
    opencode_available = False

    if opencode_path:
        try:
            result = subprocess.run(
                [opencode_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            opencode_version = result.stdout.strip() if result.returncode == 0 else None
            opencode_available = result.returncode == 0
        except Exception:
            opencode_version = None

    if claude_path:
        # Try to get version
        try:
            result = subprocess.run(
                [claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            version = "unknown"

        return {
            "claude_cli": True,
            "claude_agent_sdk": agent_sdk_available,
            "claude_path": claude_path,
            "claude_version": version,
            "chat_surface": "managed_session",
            "chat_transport": chat_transport,
            "preferred_chat_transport": preferred_transport,
            "slash_commands": _studio_supported_slash_commands(),
            "permission_profiles": _studio_supported_permission_profiles(),
            "runtime_commands": sorted(_ALLOWED_MANAGEMENT_COMMANDS["claude"]),
            "code_mode_enabled": is_code_mode_enabled(),
            "known_model_aliases": _KNOWN_CLAUDE_MODEL_ALIASES,
            "detected_default_model": detected_model.model,
            "detected_default_model_source": detected_model.source,
            "opencode_cli": opencode_available,
            "opencode_path": opencode_path,
            "opencode_version": opencode_version,
        }

    return {
        "claude_cli": False,
        "claude_agent_sdk": agent_sdk_available,
        "claude_path": None,
        "claude_version": None,
        "chat_surface": "managed_session",
        "chat_transport": chat_transport,
        "preferred_chat_transport": preferred_transport,
        "slash_commands": _studio_supported_slash_commands(),
        "permission_profiles": _studio_supported_permission_profiles(),
        "runtime_commands": sorted(_ALLOWED_MANAGEMENT_COMMANDS["claude"]),
        "fallback": "anthropic_api",
        "code_mode_enabled": is_code_mode_enabled(),
        "known_model_aliases": _KNOWN_CLAUDE_MODEL_ALIASES,
        "detected_default_model": detected_model.model,
        "detected_default_model_source": detected_model.source,
        "opencode_cli": opencode_available,
        "opencode_path": opencode_path,
        "opencode_version": opencode_version,
    }


@router.get("/studio/cwd")
async def studio_cwd():
    """Get the current working directory for Claude CLI session."""
    home = str(Path.home())
    actual_cwd = Path(os.getcwd()).resolve()
    suggested_cwd = _preferred_studio_workspace_dir(actual_cwd)

    return {
        "cwd": str(suggested_cwd),
        "actual_cwd": str(actual_cwd),
        "home": home,
        "source": "system",
        "allowed_prefixes": [str(prefix) for prefix in _allowed_workdir_prefixes()],
        "allowlist_mutation_enabled": _runtime_allowlist_mutation_enabled(),
    }
