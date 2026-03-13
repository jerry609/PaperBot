from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import agent_board as agent_board_route
from paperbot.infrastructure.swarm import CodexDispatcher, CodexResult, ReviewResult


@pytest.fixture(autouse=True)
def _isolated_board_store(monkeypatch, tmp_path):
    db_url = f"sqlite:///{tmp_path / 'agent-board-route.db'}"
    monkeypatch.setenv("PAPERBOT_DB_URL", db_url)
    store = getattr(agent_board_route, "_board_store", None)
    if store is not None:
        try:
            store.close()
        except Exception:
            pass
    agent_board_route._board_store = None
    yield
    store = getattr(agent_board_route, "_board_store", None)
    if store is not None:
        try:
            store.close()
        except Exception:
            pass
    agent_board_route._board_store = None


def _make_session_with_task(*, status: str = "human_review"):
    session = agent_board_route.BoardSession(
        session_id="board-test-session",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir="/tmp/paperbot-workspace",
    )
    task = agent_board_route.AgentTask(
        id="task-1",
        title="Implement baseline",
        description="Train baseline model",
        status=status,
        assignee="claude",
        progress=90,
        subtasks=[
            {"id": "s1", "title": "Add dataloader", "done": False},
            {"id": "s2", "title": "Train loop", "done": False},
        ],
    )
    session.tasks.append(task)
    agent_board_route._persist_session(session, checkpoint="seed", status="running")
    return task


def test_human_review_approve_marks_task_done():
    _make_session_with_task(status="human_review")

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/agent-board/tasks/task-1/human-review",
            json={"decision": "approve", "notes": "Looks good."},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "done"
    assert payload["progress"] == 100
    assert all(sub["done"] for sub in payload["subtasks"])
    assert payload["human_reviews"][-1]["decision"] == "approve"
    assert payload["human_reviews"][-1]["notes"] == "Looks good."


def test_human_review_request_changes_requeues_task():
    _make_session_with_task(status="human_review")

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/agent-board/tasks/task-1/human-review",
            json={"decision": "request_changes", "notes": "Need stronger tests."},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "planning"
    assert payload["progress"] == 0
    assert payload["human_reviews"][-1]["decision"] == "request_changes"
    assert "Need stronger tests." in (payload.get("review_feedback") or "")


def test_create_session_rejects_dangerous_workspace_dir():
    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/agent-board/sessions",
            json={
                "paper_id": "paper-1",
                "context_pack_id": "cp-1",
                "workspace_dir": "/etc",
            },
        )

    assert resp.status_code == 400
    assert "workspace_dir is not allowed" in resp.text


def test_create_session_rejects_parent_traversal_workspace_dir():
    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/agent-board/sessions",
            json={
                "paper_id": "paper-1",
                "context_pack_id": "cp-1",
                "workspace_dir": "tmp/../escape",
            },
        )

    assert resp.status_code == 400
    assert "parent traversal" in resp.text


def test_run_rejects_dangerous_workspace_dir_override():
    session = agent_board_route.BoardSession(
        session_id="board-security-test",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir=None,
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/run",
            json={"workspace_dir": "/etc"},
        )

    assert resp.status_code == 400
    assert "workspace_dir is not allowed" in resp.text


def test_session_survives_store_reinitialization():
    with TestClient(api_main.app) as client:
        create_resp = client.post(
            "/api/agent-board/sessions",
            json={
                "paper_id": "paper-1",
                "context_pack_id": "cp-1",
                "workspace_dir": "tmp/agent-board-workspace",
            },
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["session_id"]

        store = getattr(agent_board_route, "_board_store", None)
        if store is not None:
            store.close()
        agent_board_route._board_store = None

        tasks_resp = client.get(f"/api/agent-board/sessions/{session_id}/tasks")

    assert tasks_resp.status_code == 200
    assert tasks_resp.json() == []


def test_run_stream_emits_execution_logs(monkeypatch):
    class _FakeCommander:
        async def build_codex_prompt(self, task: dict, workspace):
            return f"prompt for {task.get('title')} in {workspace}"

        async def review(self, task: dict, codex_output: str):
            return ReviewResult(approved=False, feedback="Needs manual check")

        def accumulate_wisdom(self, task: dict, output: str):
            return None

    class _FakeDispatcher:
        async def dispatch(self, task_id: str, prompt: str, workspace):
            return CodexResult(
                task_id=task_id,
                success=True,
                output="generated content",
                files_generated=["src/train.py"],
            )

    session = agent_board_route.BoardSession(
        session_id="board-run-test",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir="/tmp/paperbot-workspace",
    )
    session.tasks.append(
        agent_board_route.AgentTask(
            id="task-run-1",
            title="Build trainer",
            description="Implement train.py",
            status="planning",
            assignee="claude",
            progress=0,
            subtasks=[{"id": "s1", "title": "train loop", "done": False}],
        )
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    monkeypatch.setattr(agent_board_route, "_get_commander", lambda: _FakeCommander())
    monkeypatch.setattr(agent_board_route, "_get_dispatcher", lambda: _FakeDispatcher())

    with TestClient(api_main.app) as client:
        resp = client.post(f"/api/agent-board/sessions/{session.session_id}/run", json={})

    assert resp.status_code == 200
    assert "task_reviewed" in resp.text
    assert "[DONE]" in resp.text

    updated_session = agent_board_route._load_session(session.session_id)
    assert updated_session is not None
    task = updated_session.tasks[0]
    assert task.status == "human_review"
    assert len(task.execution_log) > 0
    assert task.generated_files == ["src/train.py"]
    events = [entry["event"] for entry in task.execution_log]
    assert "task_dispatched" in events
    assert "task_codex_done" in events
    assert "task_reviewed" in events
    assert "files_written" in events


def test_run_stream_persists_generated_and_review_files_in_workspace(monkeypatch, tmp_path):
    class _FakeCommander:
        async def build_codex_prompt(self, task: dict, workspace):
            return f"prompt for {task.get('title')} in {workspace}"

        async def review(self, task: dict, codex_output: str):
            return ReviewResult(approved=False, feedback="Needs human confirmation")

        def accumulate_wisdom(self, task: dict, output: str):
            return None

    output = (
        "I added this file to isolate training flow and keep testability high.\n\n"
        "File: src/train.py\n"
        "```python\n"
        "def train_model():\n"
        '    """Run one training pass."""\n'
        "    return True\n"
        "```\n"
    )

    class _FakeCompletions:
        async def create(self, **_kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=output))]
            )

    class _FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_FakeOpenAIClient))

    dispatcher = CodexDispatcher(
        api_key="test-key",
        model="gpt-4o-mini",
        dispatch_timeout_seconds=5,
    )

    session = agent_board_route.BoardSession(
        session_id="board-run-functional",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir=None,
    )
    session.tasks.append(
        agent_board_route.AgentTask(
            id="task-functional",
            title="Build trainer",
            description="Implement train.py",
            status="planning",
            assignee="claude",
            progress=0,
            subtasks=[{"id": "s1", "title": "train loop", "done": False}],
        )
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    monkeypatch.setattr(agent_board_route, "_get_commander", lambda: _FakeCommander())
    monkeypatch.setattr(agent_board_route, "_get_dispatcher", lambda: dispatcher)

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/run",
            json={"workspace_dir": str(tmp_path)},
        )

    assert resp.status_code == 200

    updated_session = agent_board_route._load_session(session.session_id)
    assert updated_session is not None
    task = updated_session.tasks[0]
    assert task.status == "human_review"
    assert "src/train.py" in task.generated_files
    assert "reviews/task-functional-user-review.md" in task.generated_files
    assert (tmp_path / "src/train.py").exists()
    review_doc = (tmp_path / "reviews/task-functional-user-review.md").read_text(encoding="utf-8")
    assert "## What Was Added" in review_doc
    assert "## Why This Approach" in review_doc
    assert "train_model" in review_doc
