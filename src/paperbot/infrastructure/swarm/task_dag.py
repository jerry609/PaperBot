"""DAG-based task scheduler for parallel execution.

Builds a dependency graph from AgentTask list and yields execution batches.
Tasks within a batch have no mutual dependencies and can run concurrently.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from ...api.routes.agent_board import AgentTask


class TaskDAG:
    """Topological batch scheduler for AgentTask lists."""

    def __init__(self, tasks: List["AgentTask"]):
        self._tasks = {t.title: t for t in tasks}
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._in_degree: Dict[str, int] = {}

        for t in tasks:
            valid_deps = [d for d in t.dependencies if d in self._tasks]
            self._in_degree[t.title] = len(valid_deps)
            for dep in valid_deps:
                self._dependents[dep].add(t.title)

    def topological_batches(self) -> List[List["AgentTask"]]:
        """Return tasks grouped into parallel-safe batches.

        Each batch contains tasks whose dependencies are all in earlier batches.
        Falls back gracefully if cycles exist (remaining tasks appended as final batch).
        """
        in_degree = dict(self._in_degree)
        batches: List[List["AgentTask"]] = []

        ready = deque(title for title, deg in in_degree.items() if deg == 0)

        while ready:
            batch_titles = list(ready)
            ready.clear()
            batches.append([self._tasks[title] for title in batch_titles])

            for title in batch_titles:
                for dependent in self._dependents.get(title, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        ready.append(dependent)

        scheduled = sum(len(b) for b in batches)
        if scheduled < len(self._tasks):
            remaining = [self._tasks[t] for t in in_degree if in_degree[t] > 0]
            batches.append(remaining)

        return batches

    @property
    def is_trivial(self) -> bool:
        """True if all tasks are independent (no dependencies)."""
        return all(deg == 0 for deg in self._in_degree.values())
