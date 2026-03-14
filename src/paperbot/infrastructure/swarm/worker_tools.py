from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...repro.base_executor import BaseExecutor

MAX_READ_CHARS = 12_000
MAX_COMMAND_OUTPUT_CHARS = 8_000
MAX_SEARCH_OUTPUT_CHARS = 6_000
MAX_LIST_ENTRIES = 100
MAX_INSTALL_TIMEOUT_SEC = 300
MAX_COMMAND_TIMEOUT_SEC = 120
TASK_COMPLETE_SENTINEL = "TASK_COMPLETE"
INSTALL_COMMAND_PREFIXES = (
    "pip install",
    "pip3 install",
    "python -m pip install",
    "python3 -m pip install",
    "apt install",
    "apt-get install",
    "sudo apt install",
    "sudo apt-get install",
    "npm install",
    "conda install",
)


CODING_WORKER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace.",
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
            "description": "Create or overwrite a file in the workspace.",
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
                "Run a shell command inside a sandbox executor. "
                "Disabled unless CODEX_ENABLE_RUN_COMMAND=true."
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


class LocalToolExecutor:
    """Routes LLM tool calls into workspace and sandbox operations."""

    def __init__(
        self,
        workspace: Path,
        sandbox: Optional[BaseExecutor],
        task: Optional[Any] = None,
    ):
        self.workspace = workspace.resolve(strict=False)
        self.sandbox = sandbox
        self.task = task
        self.files_written: List[str] = []
        self.tool_log: List[Dict[str, Any]] = []

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if not isinstance(args, dict):
            return "Error: tool arguments must be a JSON object."

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            observation = f"Error: unknown tool '{tool_name}'."
            self._record_tool_call(tool_name, args, observation)
            return observation

        try:
            if asyncio.iscoroutinefunction(handler):
                observation = await handler(args)
            else:
                observation = handler(args)
        except Exception as exc:
            observation = f"Error: tool '{tool_name}' failed: {exc}"

        self._record_tool_call(tool_name, args, observation)
        return observation

    def _record_tool_call(self, tool_name: str, args: Dict[str, Any], observation: str) -> None:
        self.tool_log.append(
            {
                "tool": tool_name,
                "args": dict(args),
                "observation_preview": self._truncate(observation, 300),
            }
        )
        if len(self.tool_log) > 500:
            self.tool_log = self.tool_log[-500:]

    def _safe_path(self, rel_path: str) -> Optional[Path]:
        rel = (rel_path or "").strip()
        if not rel or rel == ".":
            return self.workspace

        path = Path(rel)
        if path.is_absolute() or path.drive:
            return None
        if ".." in path.parts:
            return None

        candidate = (self.workspace / path).resolve(strict=False)
        if not _is_within(candidate, self.workspace):
            return None
        return candidate

    def _tool_read_file(self, args: Dict[str, Any]) -> str:
        path = self._safe_path(str(args.get("path", "")))
        if path is None:
            return "Error: invalid path."
        if not path.exists() or not path.is_file():
            return f"File not found: {args.get('path', '')}"

        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            return (
                text[:MAX_READ_CHARS].rstrip() + "\n...[truncated]" + f"\n(total_chars={len(text)})"
            )
        return text

    def _tool_write_file(self, args: Dict[str, Any]) -> str:
        path = self._safe_path(str(args.get("path", "")))
        if path is None:
            return "Error: invalid path."

        content = str(args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        rel = path.relative_to(self.workspace).as_posix()
        if rel not in self.files_written:
            self.files_written.append(rel)
        return f"Written {len(content)} chars to {rel}"

    def _tool_list_files(self, args: Dict[str, Any]) -> str:
        target = self._safe_path(str(args.get("path", ".")))
        if target is None:
            return "Error: invalid path."
        if not target.exists():
            return f"Path not found: {args.get('path', '.')}"

        entries: List[str] = []
        if target.is_file():
            rel_file = target.relative_to(self.workspace).as_posix()
            return f"f {rel_file}"

        for p in sorted(target.rglob("*")):
            rel = p.relative_to(self.workspace).as_posix()
            if p.is_dir():
                rel = f"{rel}/"
                entries.append(f"d {rel}")
            else:
                entries.append(f"f {rel}")
            if len(entries) >= MAX_LIST_ENTRIES:
                break

        if not entries:
            return "(empty directory)"

        output = "\n".join(entries)
        if len(entries) >= MAX_LIST_ENTRIES:
            output += "\n...(truncated)"
        return output

    async def _tool_run_command(self, args: Dict[str, Any]) -> str:
        enable_run = os.getenv("CODEX_ENABLE_RUN_COMMAND", "false").lower() == "true"
        if not enable_run:
            return "Error: run_command is disabled (set CODEX_ENABLE_RUN_COMMAND=true to enable)."
        if self.sandbox is None or not self.sandbox.available():
            return "Error: run_command requires an available sandbox executor."

        command = str(args.get("command", "")).strip()
        if not command:
            return "Error: command is required."

        allowlist_error = self._check_pip_allowlist(command)
        if allowlist_error:
            return allowlist_error

        is_install = _is_install_command(command)
        timeout_sec = MAX_INSTALL_TIMEOUT_SEC if is_install else MAX_COMMAND_TIMEOUT_SEC

        result = await asyncio.to_thread(
            self.sandbox.run,
            workdir=self.workspace,
            commands=[command],
            timeout_sec=timeout_sec,
        )
        body = result.logs or ""
        if result.error:
            body = f"{body}\n[error] {result.error}".strip()
        if is_install:
            body = self._compress_install_output(body)
        body = self._truncate(body, MAX_COMMAND_OUTPUT_CHARS)
        return f"exit_code: {result.exit_code}\n{body}".strip()

    def _tool_search_files(self, args: Dict[str, Any]) -> str:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return "Error: pattern is required."

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex pattern: {exc}"

        glob = str(args.get("glob", "*")).strip() or "*"
        matches: List[str] = []

        for candidate in sorted(self.workspace.rglob(glob)):
            if not candidate.is_file():
                continue
            if not _is_within(candidate.resolve(strict=False), self.workspace):
                continue

            rel = candidate.relative_to(self.workspace).as_posix()
            try:
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

            for idx, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{idx}:{line}")
                    if sum(len(item) + 1 for item in matches) >= MAX_SEARCH_OUTPUT_CHARS:
                        return self._truncate("\n".join(matches), MAX_SEARCH_OUTPUT_CHARS)

        if not matches:
            return "(no matches)"
        return self._truncate("\n".join(matches), MAX_SEARCH_OUTPUT_CHARS)

    def _tool_update_subtask(self, args: Dict[str, Any]) -> str:
        if self.task is None:
            return "Error: no task context available."

        subtask_id = str(args.get("subtask_id", "")).strip()
        done = bool(args.get("done", False))
        notes = str(args.get("notes", "")).strip()
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
                if notes:
                    sub["notes"] = notes
                return f"Subtask '{subtask_id}' marked {'done' if done else 'not done'}."
        return f"Error: subtask '{subtask_id}' not found."

    def _tool_task_done(self, _args: Dict[str, Any]) -> str:
        return TASK_COMPLETE_SENTINEL

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated]"

    def _compress_install_output(self, output: str) -> str:
        lines = output.splitlines()
        kept: List[str] = []
        for line in lines:
            lower = line.lower().strip()
            if any(
                keyword in lower
                for keyword in (
                    "successfully installed",
                    "already satisfied",
                    "error",
                    "failed",
                    "not found",
                    "installed",
                    "collecting",
                    "warning",
                )
            ):
                kept.append(line)

        if not kept:
            return "(install completed, no notable output)"
        return "\n".join(kept[-20:])

    def _check_pip_allowlist(self, command: str) -> Optional[str]:
        allowlist_raw = os.getenv("CODEX_PIP_ALLOWLIST", "").strip()
        if not allowlist_raw:
            return None

        parsed = _parse_pip_install_packages(command)
        if parsed is None:
            return None

        allowed = {pkg.strip().lower() for pkg in allowlist_raw.split(",") if pkg.strip()}
        blocked = [pkg for pkg in parsed if pkg.lower() not in allowed]
        if blocked:
            return f"Error: packages not in allowlist: {', '.join(blocked)}"
        return None


def _is_install_command(command: str) -> bool:
    normalized = (command or "").strip().lower()
    return any(normalized.startswith(prefix) for prefix in INSTALL_COMMAND_PREFIXES)


def _parse_pip_install_packages(command: str) -> Optional[List[str]]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    if not parts:
        return []

    normalized = [item.strip() for item in parts if item.strip()]
    if normalized and normalized[0] == "sudo":
        normalized = normalized[1:]
    if not normalized:
        return []

    install_index = -1
    if len(normalized) >= 2 and normalized[0] in {"pip", "pip3"} and normalized[1] == "install":
        install_index = 1
    elif (
        len(normalized) >= 4
        and normalized[0] in {"python", "python3"}
        and normalized[1] == "-m"
        and normalized[2] in {"pip", "pip3"}
        and normalized[3] == "install"
    ):
        install_index = 3
    if install_index < 0:
        return None

    packages: List[str] = []
    for token in normalized[install_index + 1 :]:
        if token.startswith("-"):
            continue
        if token in {"&&", "||", ";", "|"}:
            break
        name = re.split(r"[<>=!~\[]", token, maxsplit=1)[0].strip().lower()
        if name:
            packages.append(name)
    return packages


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
