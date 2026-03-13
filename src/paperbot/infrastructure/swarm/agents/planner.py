"""Planner Agent -- writes structured plans to the VM's .plan/ directory.

Reads the context pack, calls Claude to decompose tasks, and writes
roadmap.md, tasks.json, context.md into /home/user/{slug}/.plan/.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..claude_commander import ClaudeCommander
from ..shared_sandbox import SharedSandbox

log = logging.getLogger(__name__)

PLAN_DIR = ".plan"


class PlannerAgent:
    """Planner Agent: decomposes context pack into tasks and writes plans to VM."""

    def __init__(self, commander: ClaudeCommander):
        self.commander = commander

    async def plan(
        self,
        sandbox: SharedSandbox,
        paper_slug: str,
        context_pack: dict,
        *,
        on_step: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        # 1. Read existing project structure from VM (may have prior work)
        existing_files = sandbox.list_files(paper_slug, ".")

        # 2. Claude decomposes into tasks
        tasks = await self.commander.decompose(context_pack)

        # 3. Write plan files directly to VM
        sandbox.run_in_paper(paper_slug, f"mkdir -p {PLAN_DIR}")
        sandbox.write_file(
            paper_slug,
            f"{PLAN_DIR}/roadmap.md",
            self._build_roadmap(context_pack, tasks, existing_files),
        )
        sandbox.write_file(
            paper_slug,
            f"{PLAN_DIR}/tasks.json",
            json.dumps(tasks, indent=2, ensure_ascii=False),
        )
        sandbox.write_file(
            paper_slug,
            f"{PLAN_DIR}/context.md",
            self._build_context_summary(context_pack),
        )

        if on_step:
            await on_step(
                "planner",
                "write_plan",
                {
                    "paper_slug": paper_slug,
                    "files": [
                        f"{PLAN_DIR}/roadmap.md",
                        f"{PLAN_DIR}/tasks.json",
                        f"{PLAN_DIR}/context.md",
                    ],
                    "tasks_count": len(tasks),
                },
            )

        return tasks

    def _build_roadmap(
        self,
        context_pack: dict,
        tasks: List[Dict[str, Any]],
        existing_files: List[str],
    ) -> str:
        lines = [
            "# Reproduction Roadmap",
            "",
            f"**Objective**: {context_pack.get('objective', 'N/A')}",
            "",
        ]
        if existing_files:
            lines.append("## Existing Files")
            for f in existing_files[:20]:
                lines.append(f"- {f}")
            lines.append("")

        lines.append("## Tasks")
        for i, t in enumerate(tasks, 1):
            title = t.get("title", "Untitled")
            desc = t.get("description", "")
            diff = t.get("difficulty", "medium")
            lines.append(f"### {i}. {title} [{diff}]")
            if desc:
                lines.append(f"{desc}")
            criteria = t.get("acceptance_criteria", [])
            if criteria:
                lines.append("**Acceptance criteria:**")
                for c in criteria:
                    lines.append(f"- {c}")
            deps = t.get("dependencies", [])
            if deps:
                lines.append(f"**Depends on:** {', '.join(deps)}")
            lines.append("")

        return "\n".join(lines)

    def _build_context_summary(self, context_pack: dict) -> str:
        lines = [
            "# Paper Context",
            "",
        ]
        paper = context_pack.get("paper", {})
        if paper:
            lines.append(f"**Title**: {paper.get('title', 'N/A')}")
            lines.append(f"**Year**: {paper.get('year', 'N/A')}")
            authors = paper.get("authors", [])
            if authors:
                lines.append(f"**Authors**: {', '.join(authors)}")
            lines.append("")

        objective = context_pack.get("objective", "")
        if objective:
            lines.append("## Objective")
            lines.append(objective)
            lines.append("")

        observations = context_pack.get("observations", [])
        if observations:
            lines.append("## Key Observations")
            for obs in observations[:10]:
                title = obs.get("title", "")
                narrative = obs.get("narrative", "")
                lines.append(f"- **{title}**: {narrative[:200]}")
            lines.append("")

        warnings = context_pack.get("warnings", [])
        if warnings:
            lines.append("## Warnings")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        return "\n".join(lines)
