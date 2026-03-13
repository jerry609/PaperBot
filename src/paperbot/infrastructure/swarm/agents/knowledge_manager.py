"""Knowledge Manager -- curates outputs in the VM's .knowledge/ directory.

Runs after all executors finish. Summarizes results, writes conventions
and learnings, cleans up temporary files.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from ..claude_commander import ClaudeCommander
from ..shared_sandbox import SharedSandbox

if TYPE_CHECKING:
    from ....api.routes.agent_board import AgentTask

log = logging.getLogger(__name__)

KNOWLEDGE_DIR = ".knowledge"


class KnowledgeManager:
    """Knowledge Manager: curates outputs in the VM after task execution."""

    def __init__(self, commander: ClaudeCommander):
        self.commander = commander

    async def curate(
        self,
        sandbox: SharedSandbox,
        paper_slug: str,
        completed_tasks: List["AgentTask"],
        *,
        on_step: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> Dict[str, str]:
        """Curate knowledge from completed tasks."""
        all_files = sandbox.list_files(paper_slug, ".")

        # Build summaries
        summary = self._build_summary(completed_tasks, all_files)
        conventions = self._build_conventions(completed_tasks)
        learnings = self._build_learnings(completed_tasks)

        # Write to VM
        sandbox.run_in_paper(paper_slug, f"mkdir -p {KNOWLEDGE_DIR}")

        written: Dict[str, str] = {}
        for name, content in [
            ("summary.md", summary),
            ("conventions.md", conventions),
            ("learnings.md", learnings),
        ]:
            path = f"{KNOWLEDGE_DIR}/{name}"
            if sandbox.write_file(paper_slug, path, content):
                written[name] = path

        # Keep .status for human review traceability and sidebar file tree.
        sandbox.run_in_paper(
            paper_slug,
            "find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        )

        # Update commander wisdom
        self._update_wisdom(completed_tasks)

        if on_step:
            await on_step(
                "knowledge_manager",
                "curate",
                {
                    "paper_slug": paper_slug,
                    "files_written": list(written.values()),
                },
            )

        return written

    def _build_summary(
        self,
        tasks: List["AgentTask"],
        all_files: List[str],
    ) -> str:
        lines = [
            "# Project Summary",
            "",
            f"Total tasks: {len(tasks)}",
            f"Successful: {sum(1 for t in tasks if t.status == 'done')}",
            f"Files in project: {len(all_files)}",
            "",
            "## Task Results",
        ]
        for t in tasks:
            status_emoji = "✅" if t.status == "done" else "❌"
            lines.append(f"- {status_emoji} **{t.title}** ({t.status})")
            if t.generated_files:
                for f in t.generated_files[:5]:
                    lines.append(f"  - {f}")
        lines.append("")
        lines.append("## Project Files")
        for f in all_files[:30]:
            lines.append(f"- {f}")

        return "\n".join(lines)

    def _build_conventions(self, tasks: List["AgentTask"]) -> str:
        lines = [
            "# Code Conventions",
            "",
            "Conventions extracted from task execution:",
            "",
        ]
        # Extract from wisdom
        if self.commander.wisdom.conventions:
            for c in self.commander.wisdom.conventions:
                lines.append(f"- {c}")
        else:
            lines.append("- (no conventions extracted yet)")
        return "\n".join(lines)

    def _build_learnings(self, tasks: List["AgentTask"]) -> str:
        lines = [
            "# Learnings",
            "",
            "Insights from task execution:",
            "",
        ]
        if self.commander.wisdom.learnings:
            for l in self.commander.wisdom.learnings:
                lines.append(f"- {l}")
        if self.commander.wisdom.gotchas:
            lines.append("")
            lines.append("## Gotchas")
            for g in self.commander.wisdom.gotchas:
                lines.append(f"- {g}")
        if not self.commander.wisdom.learnings and not self.commander.wisdom.gotchas:
            lines.append("- (no learnings extracted yet)")
        return "\n".join(lines)

    def _update_wisdom(self, tasks: List["AgentTask"]) -> None:
        for t in tasks:
            if t.status == "done" and t.codex_output:
                self.commander.accumulate_wisdom(
                    {"title": t.title, "description": t.description},
                    t.codex_output,
                )
