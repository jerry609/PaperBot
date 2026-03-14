"""Sandbox Tool Executor -- VM-native tool execution for agents.

All file operations go directly through SharedSandbox to the VM.
No local file I/O. run_command is always available since the VM
IS the workspace.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .shared_sandbox import SharedSandbox
from .worker_tools import (
    INSTALL_COMMAND_PREFIXES,
    MAX_COMMAND_OUTPUT_CHARS,
    MAX_COMMAND_TIMEOUT_SEC,
    MAX_INSTALL_TIMEOUT_SEC,
    MAX_LIST_ENTRIES,
    MAX_READ_CHARS,
    MAX_SEARCH_OUTPUT_CHARS,
    TASK_COMPLETE_SENTINEL,
)

if TYPE_CHECKING:
    from ...api.routes.agent_board import AgentTask


# Tool definitions for LLM function-calling (sandbox-as-workspace variant).
SANDBOX_WORKER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace (VM sandbox).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace (VM sandbox).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files/directories under a workspace path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command inside the VM sandbox. "
                "The workspace IS the sandbox — commands execute immediately."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents by regex pattern and optional glob.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string", "default": "*"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_subtask",
            "description": "Update subtask completion status for the current task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subtask_id": {"type": "string"},
                    "done": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
                "required": ["subtask_id", "done"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_done",
            "description": "Signal that implementation is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "files_changed": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]


def _is_install_command(command: str) -> bool:
    normalized = (command or "").strip().lower()
    return any(normalized.startswith(prefix) for prefix in INSTALL_COMMAND_PREFIXES)


def _compress_install_output(output: str) -> str:
    lines = output.splitlines()
    kept: List[str] = []
    for line in lines:
        lower = line.lower().strip()
        if any(
            kw in lower
            for kw in (
                "successfully installed", "already satisfied", "error",
                "failed", "not found", "installed", "collecting", "warning",
            )
        ):
            kept.append(line)
    if not kept:
        return "(install completed, no notable output)"
    return "\n".join(kept[-20:])


class SandboxToolExecutor:
    """Agent tool executor that routes all operations through SharedSandbox.

    All file I/O goes to the VM. run_command is always available.
    """

    def __init__(
        self,
        sandbox: SharedSandbox,
        paper_slug: str,
        task: Optional["AgentTask"] = None,
    ):
        self.sandbox = sandbox
        self.slug = paper_slug
        self.task = task
        self.files_written: List[str] = []
        self.file_snapshots: Dict[str, str] = {}  # path → content for replay
        self.tool_log: List[Dict[str, Any]] = []

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if not isinstance(args, dict):
            return "Error: tool arguments must be a JSON object."

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            observation = f"Error: unknown tool '{tool_name}'."
            self._record(tool_name, args, observation)
            return observation

        try:
            if asyncio.iscoroutinefunction(handler):
                observation = await handler(args)
            else:
                observation = handler(args)
        except Exception as exc:
            observation = f"Error: tool '{tool_name}' failed: {exc}"

        self._record(tool_name, args, observation)
        return observation

    # ----- Tool implementations -----

    def _tool_read_file(self, args: Dict[str, Any]) -> str:
        path = self._sanitize_path(str(args.get("path", "")))
        if path is None:
            return "Error: invalid path."
        if not self.sandbox.alive:
            return "Error: sandbox not available."
        content = self.sandbox.read_file(self.slug, path)
        if content is None:
            return f"File not found: {args.get('path', '')}"
        if len(content) > MAX_READ_CHARS:
            return (
                content[:MAX_READ_CHARS].rstrip()
                + "\n...[truncated]"
                + f"\n(total_chars={len(content)})"
            )
        return content

    def _tool_write_file(self, args: Dict[str, Any]) -> str:
        path = self._sanitize_path(str(args.get("path", "")))
        if path is None:
            return "Error: invalid path."
        if not self.sandbox.alive:
            return "Error: sandbox not available."
        content = str(args.get("content", ""))
        ok = self.sandbox.write_file(self.slug, path, content)
        if not ok:
            return f"Error: failed to write {path}"
        if path not in self.files_written:
            self.files_written.append(path)
        self.file_snapshots[path] = content
        return f"Written {len(content)} chars to {path}"

    def _tool_list_files(self, args: Dict[str, Any]) -> str:
        if not self.sandbox.alive:
            return "Error: sandbox not available."
        entries = self.sandbox.list_files(self.slug, args.get("path", "."))
        if not entries:
            return "(empty directory)"
        output = "\n".join(entries[:MAX_LIST_ENTRIES])
        if len(entries) > MAX_LIST_ENTRIES:
            output += "\n...(truncated)"
        return output

    async def _tool_run_command(self, args: Dict[str, Any]) -> str:
        if not self.sandbox.alive:
            return "Error: sandbox not available. Cannot execute commands."
        command = str(args.get("command", "")).strip()
        if not command:
            return "Error: command is required."

        allowlist_error = self._check_pip_allowlist(command)
        if allowlist_error:
            return allowlist_error

        is_install = _is_install_command(command)
        timeout = MAX_INSTALL_TIMEOUT_SEC if is_install else MAX_COMMAND_TIMEOUT_SEC

        result = await asyncio.to_thread(
            self.sandbox.run_in_paper, self.slug, command, timeout
        )
        body = result.logs or ""
        if result.error:
            body = f"{body}\n[error] {result.error}".strip()
        if is_install:
            body = _compress_install_output(body)
        body = self._truncate(body, MAX_COMMAND_OUTPUT_CHARS)
        return f"exit_code: {result.exit_code}\n{body}".strip()

    def _tool_search_files(self, args: Dict[str, Any]) -> str:
        if not self.sandbox.alive:
            return "Error: sandbox not available."
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return "Error: pattern is required."
        glob_pat = str(args.get("glob", "*")).strip() or "*"
        return self._truncate(
            self.sandbox.search_files(self.slug, pattern, glob_pat),
            MAX_SEARCH_OUTPUT_CHARS,
        )

    def _tool_update_subtask(self, args: Dict[str, Any]) -> str:
        if self.task is None:
            return "Error: no task context available."
        subtask_id = str(args.get("subtask_id", "")).strip()
        done = bool(args.get("done", False))
        if not subtask_id:
            return "Error: subtask_id is required."

        subtasks = getattr(self.task, "subtasks", None)
        if not isinstance(subtasks, list):
            return "Error: task has no mutable subtasks."

        for sub in subtasks:
            if not isinstance(sub, dict):
                continue
            if str(sub.get("id", "")).strip() == subtask_id:
                sub["done"] = done
                notes = str(args.get("notes", "")).strip()
                if notes:
                    sub["notes"] = notes
                return f"Subtask '{subtask_id}' marked {'done' if done else 'not done'}."
        return f"Error: subtask '{subtask_id}' not found."

    def _tool_task_done(self, _args: Dict[str, Any]) -> str:
        return TASK_COMPLETE_SENTINEL

    # ----- Path safety -----

    def _sanitize_path(self, rel: str) -> Optional[str]:
        rel = (rel or "").strip()
        if not rel:
            return None
        path = Path(rel)
        if path.is_absolute():
            return None
        if ".." in path.parts:
            return None
        parts = [p for p in path.parts if p not in ("", ".")]
        if not parts:
            return None
        return str(Path(*parts))

    # ----- Utilities -----

    def _check_pip_allowlist(self, command: str) -> Optional[str]:
        allowlist_raw = os.getenv("CODEX_PIP_ALLOWLIST", "").strip()
        if not allowlist_raw:
            return None
        # Only check pip install commands
        normalized = (command or "").strip().lower()
        if not any(normalized.startswith(p) for p in ("pip install", "pip3 install")):
            return None
        try:
            parts = shlex.split(command)
        except ValueError:
            return None
        allowed = {pkg.strip().lower() for pkg in allowlist_raw.split(",") if pkg.strip()}
        packages = [
            p for p in parts[2:]
            if p and not p.startswith("-") and p not in ("&&", "||", ";", "|")
        ]
        blocked = [p for p in packages if re.split(r"[<>=!~\[]", p)[0].lower() not in allowed]
        if blocked:
            return f"Error: packages not in allowlist: {', '.join(blocked)}"
        return None

    def _record(self, tool_name: str, args: Dict[str, Any], observation: str) -> None:
        self.tool_log.append({
            "tool": tool_name,
            "args": dict(args),
            "observation_preview": self._truncate(observation, 300),
        })
        if len(self.tool_log) > 500:
            self.tool_log = self.tool_log[-500:]

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated]"
