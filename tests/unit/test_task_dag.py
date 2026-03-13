"""Tests for TaskDAG -- topological batch scheduler."""

import pytest
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


# Minimal AgentTask stub matching the real model's fields used by TaskDAG
class _AgentTask(BaseModel):
    id: str
    title: str
    description: str = ""
    status: Literal["planning", "in_progress", "done"] = "planning"
    assignee: str = "claude"
    progress: int = 0
    tags: List[str] = []
    subtasks: List[Dict[str, Any]] = []
    paper_id: Optional[str] = None
    codex_output: Optional[str] = None
    review_feedback: Optional[str] = None
    generated_files: List[str] = Field(default_factory=list)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    human_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


# Monkeypatch: TaskDAG uses TYPE_CHECKING import for AgentTask
# so at runtime it only uses .title and .dependencies attributes.
from paperbot.infrastructure.swarm.task_dag import TaskDAG


def _task(tid: str, title: str, deps: List[str] = None) -> _AgentTask:
    return _AgentTask(id=tid, title=title, dependencies=deps or [])


def test_all_independent():
    """No dependencies → single batch with all tasks."""
    tasks = [_task("1", "A"), _task("2", "B"), _task("3", "C")]
    dag = TaskDAG(tasks)
    batches = dag.topological_batches()
    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert dag.is_trivial


def test_linear_chain():
    """A → B → C produces 3 batches of 1."""
    tasks = [
        _task("1", "A"),
        _task("2", "B", ["A"]),
        _task("3", "C", ["B"]),
    ]
    dag = TaskDAG(tasks)
    batches = dag.topological_batches()
    assert len(batches) == 3
    assert [b[0].title for b in batches] == ["A", "B", "C"]
    assert not dag.is_trivial


def test_diamond():
    """A → B, A → C, B → D, C → D → batches: [A], [B,C], [D]."""
    tasks = [
        _task("1", "A"),
        _task("2", "B", ["A"]),
        _task("3", "C", ["A"]),
        _task("4", "D", ["B", "C"]),
    ]
    dag = TaskDAG(tasks)
    batches = dag.topological_batches()
    assert len(batches) == 3
    assert batches[0][0].title == "A"
    assert sorted(t.title for t in batches[1]) == ["B", "C"]
    assert batches[2][0].title == "D"


def test_unknown_dependency_ignored():
    """Dependencies on non-existent tasks are silently ignored."""
    tasks = [
        _task("1", "A", ["NonExistent"]),
        _task("2", "B"),
    ]
    dag = TaskDAG(tasks)
    batches = dag.topological_batches()
    # Both should be in batch 0 since "NonExistent" is ignored
    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_cycle_fallback():
    """Circular dependency → remaining tasks appended as final batch."""
    tasks = [
        _task("1", "A", ["C"]),
        _task("2", "B", ["A"]),
        _task("3", "C", ["B"]),
        _task("4", "D"),  # independent
    ]
    dag = TaskDAG(tasks)
    batches = dag.topological_batches()
    # D should be in first batch (no deps); A,B,C form a cycle
    total_scheduled = sum(len(b) for b in batches)
    assert total_scheduled == 4  # all tasks present (no crash)
    # D is independent, should appear
    all_titles = [t.title for b in batches for t in b]
    assert "D" in all_titles


def test_single_task():
    """Single task → single batch."""
    tasks = [_task("1", "Setup")]
    dag = TaskDAG(tasks)
    batches = dag.topological_batches()
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert dag.is_trivial


def test_wide_fan_out():
    """One root task with many dependents → 2 batches."""
    root = _task("0", "Root")
    children = [_task(str(i), f"Child-{i}", ["Root"]) for i in range(1, 6)]
    dag = TaskDAG([root] + children)
    batches = dag.topological_batches()
    assert len(batches) == 2
    assert batches[0][0].title == "Root"
    assert len(batches[1]) == 5
