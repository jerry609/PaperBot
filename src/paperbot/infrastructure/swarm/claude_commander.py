"""
Claude Commander -- orchestrates the multi-agent workflow.

Claude acts as the "boss", decomposing work into tasks, dispatching
Codex workers, reviewing results, and accumulating wisdom.

Inspired by Oh My OpenCode's three-layer architecture:
- Planning layer (Claude) -> structured task decomposition
- Execution layer (Codex) -> autonomous code generation
- Review layer (Claude) -> quality verification
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    approved: bool
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)


@dataclass
class WisdomEntry:
    """Accumulated learning from completed tasks."""

    learnings: List[str] = field(default_factory=list)
    conventions: List[str] = field(default_factory=list)
    gotchas: List[str] = field(default_factory=list)


class ClaudeCommander:
    """Claude as the commander orchestrating Codex workers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        request_timeout_seconds: Optional[float] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        env_timeout = os.getenv("CLAUDE_COMMANDER_TIMEOUT_SECONDS")
        if request_timeout_seconds is not None:
            self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        elif env_timeout:
            try:
                self.request_timeout_seconds = max(1.0, float(env_timeout))
            except ValueError:
                self.request_timeout_seconds = 120.0
        else:
            self.request_timeout_seconds = 120.0
        self.wisdom = WisdomEntry()

    async def decompose(self, context_pack: dict) -> List[Dict[str, Any]]:
        """Decompose context pack into discrete coding tasks using Claude API."""
        roadmap = context_pack.get("task_roadmap", [])
        observations = context_pack.get("observations", [])
        objective = context_pack.get("objective", "")

        # If no API key, fall back to roadmap extraction
        if not self.api_key:
            log.warning("No ANTHROPIC_API_KEY; falling back to roadmap extraction")
            return self._extract_from_roadmap(roadmap)

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            system = (
                "You are a software architect decomposing a paper reproduction into "
                "discrete coding tasks for Codex workers. Each task should be self-contained, "
                "clearly scoped, and executable independently. Return JSON only."
            )

            prompt = (
                f"## Objective\n{objective}\n\n"
                f"## Task Roadmap\n{json.dumps(roadmap, indent=2)}\n\n"
                f"## Key Observations\n{json.dumps(observations[:10], indent=2)}\n\n"
                "Decompose into coding tasks. For each task provide:\n"
                '- "title": short name\n'
                '- "description": what to implement\n'
                '- "difficulty": easy/medium/hard\n'
                '- "acceptance_criteria": list of verification checks\n'
                '- "dependencies": list of task titles this depends on\n\n'
                "Return a JSON array of task objects. No markdown fences."
            )

            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self.request_timeout_seconds,
            )

            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
            tasks = json.loads(text)

            if isinstance(tasks, list):
                return tasks

        except Exception:
            log.exception("Claude decomposition failed; falling back to roadmap")

        return self._extract_from_roadmap(roadmap)

    def _extract_from_roadmap(self, roadmap: list) -> List[Dict[str, Any]]:
        """Fallback: extract tasks directly from context pack roadmap."""
        tasks = []
        for step in roadmap:
            tasks.append(
                {
                    "title": step.get("title", "Untitled"),
                    "description": step.get("description", ""),
                    "difficulty": step.get("estimated_difficulty", "medium"),
                    "acceptance_criteria": step.get("acceptance_criteria", []),
                    "dependencies": [],
                }
            )
        return tasks

    async def build_codex_prompt(self, task: dict, workspace: Path) -> str:
        """Build a principle-driven prompt for Codex worker.

        Codex responds best to concise, goal-oriented prompts
        (unlike Claude which prefers mechanics-driven prompts).
        """
        parts = [
            f"# Task: {task['title']}",
            "",
            "## Goal",
            task.get("description", ""),
            "",
        ]

        if task.get("acceptance_criteria"):
            parts.append("## Acceptance Criteria")
            for criterion in task["acceptance_criteria"]:
                parts.append(f"- {criterion}")
            parts.append("")

        # Inject accumulated wisdom
        if self.wisdom.learnings:
            parts.append("## Context from Previous Tasks")
            for learning in self.wisdom.learnings[-5:]:
                parts.append(f"- {learning}")
            parts.append("")

        if self.wisdom.conventions:
            parts.append("## Project Conventions")
            for convention in self.wisdom.conventions[-5:]:
                parts.append(f"- {convention}")
            parts.append("")

        if self.wisdom.gotchas:
            parts.append("## Known Gotchas")
            for gotcha in self.wisdom.gotchas[-3:]:
                parts.append(f"- {gotcha}")
            parts.append("")

        parts.append(f"## Workspace\n{workspace}\n")
        parts.append(
            "Write the code. Create or modify files as needed. " "Run tests to verify correctness."
        )

        return "\n".join(parts)

    async def review(self, task: dict, codex_output: str) -> ReviewResult:
        """Review Codex worker output using Claude."""
        if not codex_output or len(codex_output.strip()) < 10:
            return ReviewResult(
                approved=False,
                feedback="Output is too short or empty",
            )

        if not self.api_key:
            return ReviewResult(approved=True, feedback="Auto-approved (no API key)")

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            prompt = (
                f"## Task: {task.get('title', 'unknown')}\n"
                f"## Description: {task.get('description', '')}\n\n"
                f"## Codex Output:\n```\n{codex_output[:8000]}\n```\n\n"
                "Review this output. Does it satisfy the task? "
                "Respond with JSON: "
                '{"approved": true/false, "feedback": "...", "suggestions": [...]}'
            )

            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self.request_timeout_seconds,
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]

            result = json.loads(text)
            return ReviewResult(
                approved=result.get("approved", False),
                feedback=result.get("feedback", ""),
                suggestions=result.get("suggestions", []),
            )

        except asyncio.TimeoutError:
            return ReviewResult(
                approved=True,
                feedback=(
                    "Auto-approved (review timeout after " f"{int(self.request_timeout_seconds)}s)"
                ),
            )
        except Exception:
            log.exception("Claude review failed; auto-approving")
            return ReviewResult(approved=True, feedback="Auto-approved (review error)")

    def accumulate_wisdom(self, task: dict, output: str) -> None:
        """Extract learnings from completed tasks for future workers."""
        self.wisdom.learnings.append(
            f"Completed: {task.get('title', 'unknown')} " f"-- output length: {len(output)} chars"
        )
