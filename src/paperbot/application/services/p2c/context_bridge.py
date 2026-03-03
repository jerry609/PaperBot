"""ContextEngineBridge — injects user memory and project context into NormalizedInput."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from paperbot.application.services.p2c.models import NormalizedInput

logger = logging.getLogger(__name__)

_MAX_MEMORY_ITEMS = 5
_MAX_TASKS = 3


class ContextEngineBridge:
    """
    Enriches NormalizedInput with user memory and project context from ContextEngine,
    so that P2C extraction stages can personalise their output.

    Gracefully degrades: if ContextEngine is unavailable or returns no data,
    normalized_input is returned unchanged.
    """

    def __init__(self, engine=None) -> None:
        self._engine = engine

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from paperbot.context_engine.engine import ContextEngine

            return ContextEngine()
        except Exception:
            logger.debug("ContextEngine unavailable, skipping context enrichment")
            return None

    async def enrich(
        self,
        normalized_input: NormalizedInput,
        *,
        user_id: str,
        track_id: Optional[int] = None,
    ) -> NormalizedInput:
        """
        Query ContextEngine for user memory and track goals, then inject them into
        normalized_input.user_memory and normalized_input.project_context.

        Returns the same NormalizedInput object (mutated in place).
        """
        if user_id == "default":
            return normalized_input

        engine = self._get_engine()
        if engine is None:
            return normalized_input

        try:
            query = f"{normalized_input.paper.title} {normalized_input.abstract[:200]}"
            context_pack = await engine.build_context_pack(
                user_id=user_id,
                query=query,
                track_id=track_id,
            )
            user_memory = _format_user_memory(context_pack)
            project_context = _format_project_context(context_pack)
            if user_memory:
                normalized_input.user_memory = user_memory
            if project_context:
                normalized_input.project_context = project_context
        except Exception:
            logger.warning(
                "ContextEngineBridge.enrich failed, continuing without user context",
                exc_info=True,
            )

        return normalized_input


def _format_user_memory(context_pack: Dict[str, Any]) -> Optional[str]:
    """Combine global user preferences and track memories into a compact bullet list."""
    seen: set[str] = set()
    lines: List[str] = []

    for m in context_pack.get("user_prefs", [])[:_MAX_MEMORY_ITEMS]:
        content = (m.get("content") or "").strip()
        if content and content not in seen:
            lines.append(f"- {content}")
            seen.add(content)

    for m in context_pack.get("relevant_memories", [])[:_MAX_MEMORY_ITEMS]:
        content = (m.get("content") or "").strip()
        if content and content not in seen:
            lines.append(f"- {content}")
            seen.add(content)

    return "\n".join(lines) if lines else None


def _format_project_context(context_pack: Dict[str, Any]) -> Optional[str]:
    """Format the active track and in-progress tasks into a compact text block."""
    parts: List[str] = []

    track = context_pack.get("active_track")
    if track:
        name = (track.get("name") or "").strip()
        goal = (track.get("goal") or "").strip()
        if name:
            parts.append(f"Research track: {name}")
        if goal:
            parts.append(f"Goal: {goal}")

    tasks = context_pack.get("progress_state", {}).get("tasks", [])
    active_tasks = [t for t in tasks if t.get("status") == "active"][:_MAX_TASKS]
    if active_tasks:
        task_lines = [
            f"  - {t.get('title', '').strip()}" for t in active_tasks if t.get("title")
        ]
        if task_lines:
            parts.append("Active tasks:\n" + "\n".join(task_lines))

    return "\n".join(parts) if parts else None
