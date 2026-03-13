"""Executor Agent -- implements code directly inside the VM.

Reads .plan/ → implements code → writes .status/{task_id}.json.
All operations go through SandboxToolExecutor (VM-native).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from ..codex_dispatcher import CodexDispatcher, CodexResult
from ..sandbox_tool_executor import SandboxToolExecutor
from ..shared_sandbox import SharedSandbox

if TYPE_CHECKING:
    from ....api.routes.agent_board import AgentTask

log = logging.getLogger(__name__)

STATUS_DIR = ".status"


class ExecutorAgent:
    """Executor Sub-Agent: implements code in the VM sandbox.

    The agent writes, runs, and observes inside the same VM.
    Zero upload delay — write_file → run_command is instant.
    """

    def __init__(self, dispatcher: CodexDispatcher):
        self.dispatcher = dispatcher

    async def execute(
        self,
        task: "AgentTask",
        sandbox: SharedSandbox,
        paper_slug: str,
        *,
        on_step: Optional[Callable[..., Awaitable[None]]] = None,
        on_think: Optional[Callable[[int, str], Awaitable[None]]] = None,
        wisdom: Optional[List[str]] = None,
        max_iterations: int = 25,
    ) -> CodexResult:
        # 1. Read plan and prior status from VM
        plan = sandbox.read_file(paper_slug, ".plan/roadmap.md") or ""
        context = sandbox.read_file(paper_slug, ".plan/context.md") or ""
        prior = self._read_prior_status(sandbox, paper_slug)

        # 2. Build prompt
        prompt = self._build_prompt(task, plan, context, prior, wisdom=wisdom)

        # 3. Create VM-native tool executor
        tool_exec = SandboxToolExecutor(sandbox, paper_slug, task)

        # 4. Run CodeAct tool loop
        result = await self.dispatcher.dispatch_with_sandbox_tools(
            task_id=task.id,
            prompt=prompt,
            tool_executor=tool_exec,
            on_step=on_step,
            on_think=on_think,
            max_iterations=max_iterations,
        )

        # 5. Write status file to VM
        sandbox.run_in_paper(paper_slug, f"mkdir -p {STATUS_DIR}")
        sandbox.write_file(
            paper_slug,
            f"{STATUS_DIR}/{task.id}.json",
            json.dumps(
                {
                    "task_id": task.id,
                    "title": task.title,
                    "success": result.success,
                    "files_generated": tool_exec.files_written,
                    "summary": (result.output or "")[:1000],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        # Merge written files into result
        result.files_generated = list(
            dict.fromkeys([*result.files_generated, *tool_exec.files_written])
        )
        result.file_snapshots = dict(tool_exec.file_snapshots)

        return result

    def _read_prior_status(self, sandbox: SharedSandbox, slug: str) -> str:
        # Read full task list from .plan/tasks.json for scope awareness
        tasks_json_raw = sandbox.read_file(slug, ".plan/tasks.json")
        all_tasks: List[Dict[str, Any]] = []
        if tasks_json_raw:
            try:
                all_tasks = json.loads(tasks_json_raw)
            except Exception:
                pass

        # Read completed status from .status/
        entries = sandbox.list_files(slug, STATUS_DIR)
        completed: Dict[str, Dict[str, Any]] = {}
        for f in entries:
            if not f.endswith(".json"):
                continue
            content = sandbox.read_file(slug, f"{STATUS_DIR}/{f}")
            if not content:
                continue
            try:
                d = json.loads(content)
                completed[d.get("title", "")] = d
            except Exception:
                pass

        # If we have the full task list, show status for each
        if all_tasks:
            lines: List[str] = []
            for t in all_tasks:
                title = t.get("title", "")
                if title in completed:
                    d = completed[title]
                    marker = "\u2713" if d.get("success") else "\u2717"
                    lines.append(
                        f"- [{marker}] {title}: {d.get('summary', '')[:100]}"
                    )
                else:
                    lines.append(f"- [pending] {title}")
            return "\n".join(lines) if lines else "(no prior tasks)"

        # Fallback: only show completed tasks
        lines = []
        for d in completed.values():
            marker = "\u2713" if d.get("success") else "\u2717"
            lines.append(
                f"- [{marker}] {d.get('title', '?')}: "
                f"{d.get('summary', '')[:100]}"
            )
        return "\n".join(lines) if lines else "(no prior tasks)"

    def _build_prompt(
        self,
        task: "AgentTask",
        plan: str,
        context: str,
        prior_status: str,
        *,
        wisdom: Optional[List[str]] = None,
    ) -> str:
        parts = [
            f"# Task: {task.title}",
            "",
            "## Goal",
            task.description,
            "",
        ]

        if task.subtasks:
            parts.append("## Subtasks (Acceptance Criteria)")
            for sub in task.subtasks:
                if not isinstance(sub, dict):
                    continue
                sid = str(sub.get("id", "subtask"))
                title = str(sub.get("title", "")).strip()
                done = bool(sub.get("done", False))
                marker = "x" if done else " "
                parts.append(f"- [{marker}] {sid}: {title}")
            parts.append("")
            parts.append("Call update_subtask whenever you complete one criterion.")
            parts.append("")

        if plan:
            parts.append("## Project Roadmap (from .plan/roadmap.md)")
            parts.append(plan[:2000])
            parts.append("")

        if context:
            parts.append("## Paper Context (from .plan/context.md)")
            parts.append(context[:1500])
            parts.append("")

        if prior_status and prior_status != "(no prior tasks)":
            parts.append("## Prior Tasks Status")
            parts.append(prior_status)
            parts.append("")

        if wisdom:
            parts.append("## Context from Previous Tasks")
            for learning in wisdom[-5:]:
                parts.append(f"- {learning}")
            parts.append("")

        parts.append(
            "## Instructions\n"
            "You are working directly inside a VM sandbox. "
            "All files you write are immediately available for execution.\n\n"
            "**Think out loud**: Before every tool call, write a brief 'Thought:' explaining "
            "what you are about to do and why. This makes your reasoning visible.\n\n"
            "**File organization**: Place code in a proper directory structure:\n"
            "- src/ or <project_name>/ for main source modules\n"
            "- tests/ for test files\n"
            "- configs/ for configuration files\n"
            "- Entry point (main.py, train.py) in root; implementation in subdirectories\n"
            "- Never dump all files flat in root — group related code into packages\n\n"
            "Steps:\n"
            "1. Use list_files/read_file to understand existing code.\n"
            "2. Use write_file to implement changes.\n"
            "3. Use run_command to verify behavior (e.g. python main.py, pytest -q).\n"
            "4. Update progress with update_subtask.\n"
            "5. Call task_done with a short summary when complete.\n\n"
            "Rules:\n"
            "- Make minimal, correct changes.\n"
            "- Inspect outputs before taking next action.\n"
            "- If a command fails, diagnose and fix before continuing.\n"
            "- Do not reinstall packages that are already available."
        )

        return "\n".join(parts)
