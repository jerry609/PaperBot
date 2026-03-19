from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import agent_board as agent_board_route
from paperbot.infrastructure.swarm import CodexDispatcher, CodexResult, ReviewResult
from paperbot.repro.execution_result import ExecutionResult


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


def test_create_session_persists_user_id():
    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/agent-board/sessions",
            json={
                "paper_id": "paper-1",
                "context_pack_id": "cp-1",
                "workspace_dir": "tmp/agent-board-workspace",
                "user_id": "user-42",
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["user_id"] == "user-42"
    assert "sandbox_id" in payload
    assert payload["lifecycle_events"]
    assert payload["lifecycle_events"][0]["event"] == "session_created"


def test_get_session_endpoint_returns_persisted_snapshot():
    session = agent_board_route.BoardSession(
        session_id="board-session-get",
        paper_id="paper-get-1",
        context_pack_id="cp-get-1",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-get",
    )
    session.tasks.append(
        agent_board_route.AgentTask(
            id="task-get-1",
            title="Implement loader",
            description="Load data",
            status="planning",
            assignee="claude",
            progress=0,
            subtasks=[{"id": "s1", "title": "step", "done": False}],
        )
    )
    agent_board_route._persist_session(session, checkpoint="planned", status="running")

    with TestClient(api_main.app) as client:
        resp = client.get(f"/api/agent-board/sessions/{session.session_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == session.session_id
    assert payload["paper_id"] == "paper-get-1"
    assert payload["context_pack_id"] == "cp-get-1"
    assert payload["tasks"][0]["id"] == "task-get-1"
    assert payload["status"] == "running"
    assert payload["checkpoint"] == "planned"


def test_get_session_endpoint_returns_404_for_missing_session():
    with TestClient(api_main.app) as client:
        resp = client.get("/api/agent-board/sessions/does-not-exist")

    assert resp.status_code == 404


def test_create_task_endpoint_persists_studio_ad_hoc_task():
    session = agent_board_route.BoardSession(
        session_id="board-session-create-task",
        paper_id="paper-create-task",
        context_pack_id="cp-create-task",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-create-task",
    )
    agent_board_route._persist_session(session, checkpoint="created", status="running")

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/tasks",
            json={
                "title": "Implement Studio Codex bridge",
                "description": "Create a real Codex subagent task from Studio console.",
                "assignee": "codex",
                "tags": ["monitor"],
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["title"] == "Implement Studio Codex bridge"
    assert payload["description"] == "Create a real Codex subagent task from Studio console."
    assert payload["status"] == "planning"
    assert payload["assignee"] == "claude"
    assert "studio" in payload["tags"]
    assert "ad_hoc" in payload["tags"]
    assert "monitor" in payload["tags"]

    persisted = agent_board_route._load_session(session.session_id)
    assert persisted is not None
    assert len(persisted.tasks) == 1
    assert persisted.tasks[0].id == payload["id"]


def test_create_task_endpoint_rejects_opencode_runtime():
    session = agent_board_route.BoardSession(
        session_id="board-session-opencode",
        paper_id="paper-opencode",
        context_pack_id="cp-opencode",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-opencode",
    )
    agent_board_route._persist_session(session, checkpoint="created", status="running")

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/tasks",
            json={
                "title": "Try OpenCode",
                "description": "Attempt to route Studio delegation into OpenCode.",
                "assignee": "opencode",
            },
        )

    assert resp.status_code == 400
    assert "Only Codex delegation is wired right now" in resp.text


def test_get_latest_session_by_paper_returns_latest_match():
    first = agent_board_route.BoardSession(
        session_id="board-session-latest-1",
        paper_id="paper-latest",
        context_pack_id="cp-latest-1",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-latest",
    )
    second = agent_board_route.BoardSession(
        session_id="board-session-latest-2",
        paper_id="paper-latest",
        context_pack_id="cp-latest-2",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-latest",
    )
    other = agent_board_route.BoardSession(
        session_id="board-session-other-paper",
        paper_id="paper-other",
        context_pack_id="cp-other",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-latest",
    )
    agent_board_route._persist_session(first, checkpoint="created", status="running")
    agent_board_route._persist_session(other, checkpoint="created", status="running")
    agent_board_route._persist_session(second, checkpoint="created", status="running")

    with TestClient(api_main.app) as client:
        resp = client.get("/api/agent-board/sessions/latest/by-paper", params={"paper_id": "paper-latest"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == "board-session-latest-2"
    assert payload["paper_id"] == "paper-latest"
    assert payload["context_pack_id"] == "cp-latest-2"
    assert payload["found"] is True


def test_get_latest_session_by_paper_returns_empty_payload_when_absent():
    with TestClient(api_main.app) as client:
        resp = client.get("/api/agent-board/sessions/latest/by-paper", params={"paper_id": "paper-missing"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["found"] is False
    assert payload["session_id"] is None
    assert payload["paper_id"] == "paper-missing"
    assert payload["tasks"] == []


def test_get_session_sandbox_sets_sandbox_id(monkeypatch):
    class _FakeSandbox:
        executor_type = "E2BExecutor"
        sandbox_id = "sbx-123"

        def available(self):
            return True

    class _FakeManager:
        def get_or_create(self, **_kwargs):
            return _FakeSandbox(), "sbx-123"

    session = agent_board_route.BoardSession(
        session_id="board-sbx-test",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="u1",
    )

    monkeypatch.setattr(agent_board_route, "_sandbox_manager", _FakeManager())
    sandbox = agent_board_route._get_session_sandbox(session)

    assert sandbox is not None
    assert session.sandbox_id == "sbx-123"
    assert session.sandbox_executor == "E2BExecutor"
    assert session.lifecycle_events
    assert session.lifecycle_events[-1]["event"] == "sandbox_attached"


def test_get_session_sandbox_endpoint_returns_metadata(monkeypatch):
    class _FakeSandbox:
        executor_type = "E2BExecutor"
        sandbox_id = "sbx-endpoint"

        def available(self):
            return True

    class _FakeManager:
        def get_or_create(self, **_kwargs):
            return _FakeSandbox(), "sbx-endpoint"

        def lease_for_user(self, user_key: str):
            return types.SimpleNamespace(
                user_key=user_key,
                session_id="board-sandbox-endpoint",
                executor_type="E2BExecutor",
                sandbox_id="sbx-endpoint",
                updated_at="2026-03-10T00:00:00+00:00",
            )

    session = agent_board_route.BoardSession(
        session_id="board-sandbox-endpoint",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="u1",
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    monkeypatch.setattr(agent_board_route, "_sandbox_manager", _FakeManager())

    with TestClient(api_main.app) as client:
        resp = client.get(f"/api/agent-board/sessions/{session.session_id}/sandbox?resolve=true")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["mode"] == "persistent"
    assert payload["sandbox"]["active"] is True
    assert payload["sandbox"]["sandbox_id"] == "sbx-endpoint"
    assert payload["sandbox"]["executor_type"] == "E2BExecutor"

    updated = agent_board_route._load_session(session.session_id)
    assert updated is not None
    assert updated.sandbox_id == "sbx-endpoint"


def test_list_sandbox_tree_endpoint_returns_recursive_files(monkeypatch):
    session = agent_board_route.BoardSession(
        session_id="board-sandbox-tree",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="u-tree",
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    class _FakeSharedSandbox:
        alive = True

        def list_files_recursive(self, slug: str):
            assert slug == session.paper_slug_name
            return [
                "src/main.py",
                ".plan/roadmap.md",
                ".knowledge/summary.md",
            ]

    monkeypatch.setattr(agent_board_route, "_get_shared_sandbox", lambda _session: _FakeSharedSandbox())

    with TestClient(api_main.app) as client:
        resp = client.get(f"/api/agent-board/sessions/{session.session_id}/sandbox/tree")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["paper_slug"] == session.paper_slug_name
    assert payload["files"] == [
        ".knowledge/summary.md",
        ".plan/roadmap.md",
        "src/main.py",
    ]


def test_release_session_sandbox_clears_all_user_sessions(monkeypatch):
    class _FakeManager:
        def __init__(self):
            self.terminated: list[str] = []

        def lease_for_user(self, user_key: str):
            return types.SimpleNamespace(
                user_key=user_key,
                session_id="board-release-1",
                executor_type="E2BExecutor",
                sandbox_id="sbx-user",
                updated_at="2026-03-10T00:00:00+00:00",
            )

        def terminate(self, *, user_key: str):
            self.terminated.append(user_key)

    session_one = agent_board_route.BoardSession(
        session_id="board-release-1",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-a",
        sandbox_id="sbx-user",
        sandbox_executor="E2BExecutor",
    )
    session_two = agent_board_route.BoardSession(
        session_id="board-release-2",
        paper_id="paper-2",
        context_pack_id="cp-2",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-a",
        sandbox_id="sbx-user",
        sandbox_executor="E2BExecutor",
    )
    agent_board_route._persist_session(session_one, checkpoint="seed", status="running")
    agent_board_route._persist_session(session_two, checkpoint="seed", status="running")

    fake_manager = _FakeManager()
    monkeypatch.setattr(agent_board_route, "_sandbox_manager", fake_manager)

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session_one.session_id}/sandbox/release",
            json={"reason": "manual-test"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["released"] is True
    assert payload["cleared_sessions"] == 2
    assert fake_manager.terminated == ["user-a"]

    updated_one = agent_board_route._load_session(session_one.session_id)
    updated_two = agent_board_route._load_session(session_two.session_id)
    assert updated_one is not None
    assert updated_two is not None
    assert updated_one.sandbox_id is None
    assert updated_two.sandbox_id is None
    assert any(e["event"] == "sandbox_released" for e in updated_one.lifecycle_events)
    assert any(e["event"] == "sandbox_released" for e in updated_two.lifecycle_events)


def test_archive_session_with_release_marks_completed(monkeypatch):
    class _FakeManager:
        def lease_for_user(self, user_key: str):
            return types.SimpleNamespace(
                user_key=user_key,
                session_id="board-archive-1",
                executor_type="E2BExecutor",
                sandbox_id="sbx-archive",
                updated_at="2026-03-10T00:00:00+00:00",
            )

        def terminate(self, *, user_key: str):
            return None

    session = agent_board_route.BoardSession(
        session_id="board-archive-1",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir="/tmp/paperbot-workspace",
        user_id="user-archive",
        sandbox_id="sbx-archive",
        sandbox_executor="E2BExecutor",
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")
    monkeypatch.setattr(agent_board_route, "_sandbox_manager", _FakeManager())

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/archive",
            json={"release_sandbox": True, "reason": "done"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "completed"
    assert payload["checkpoint"] == "archived"
    assert payload["release"]["released"] is True
    assert payload["session"]["sandbox_id"] is None

    row = agent_board_route._get_board_store().get_session(session.session_id)
    assert row is not None
    assert row["status"] == "completed"
    assert row["checkpoint"] == "archived"


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
    monkeypatch.setenv("PAPERBOT_SANDBOX_WORKSPACE", "false")

    class _FakeCommander:
        async def build_codex_prompt(self, task: dict, workspace):
            return f"prompt for {task.get('title')} in {workspace}"

        async def review(self, task: dict, codex_output: str):
            return ReviewResult(approved=False, feedback="Needs manual check")

        def accumulate_wisdom(self, task: dict, output: str):
            return None

    class _FakeDispatcher:
        async def dispatch_auto(self, task_id: str, prompt: str, workspace, **_kwargs):
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
    monkeypatch.setenv("CODEX_TOOL_USE", "false")

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


def test_run_stream_sandbox_verify_failure_then_repair_success(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERBOT_SANDBOX_WORKSPACE", "false")

    class _FakeCommander:
        async def build_codex_prompt(self, task: dict, workspace):
            return f"prompt for {task.get('title')} in {workspace}"

        async def build_codex_repair_prompt(
            self, task: dict, workspace, *, verify_summary: str, attempt: int
        ):
            return (
                f"repair prompt for {task.get('title')} in {workspace} "
                f"(attempt={attempt}, verify={verify_summary})"
            )

        async def review(self, task: dict, codex_output: str):
            return ReviewResult(approved=True, feedback="Looks correct")

        def accumulate_wisdom(self, task: dict, output: str):
            return None

    class _FakeDispatcher:
        def __init__(self):
            self.calls = 0

        async def dispatch_auto(self, task_id: str, prompt: str, workspace, **_kwargs):
            self.calls += 1
            return CodexResult(
                task_id=task_id,
                success=True,
                output=f"generated content call {self.calls}",
                files_generated=["src/train.py"],
            )

    class _FakeSandbox:
        executor_type = "FakeSandbox"

        def __init__(self):
            self._results = [
                ExecutionResult(status="failed", exit_code=1, logs="initial verify failed"),
                ExecutionResult(status="success", exit_code=0, logs="verify passed after repair"),
            ]

        def available(self) -> bool:
            return True

        def run(self, workdir, commands, timeout_sec=300, cache_dir=None, record_meta=True):
            if self._results:
                return self._results.pop(0)
            return ExecutionResult(status="success", exit_code=0, logs="default pass")

    session = agent_board_route.BoardSession(
        session_id="board-run-sandbox-repair",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir=str(tmp_path),
    )
    session.tasks.append(
        agent_board_route.AgentTask(
            id="task-sandbox-repair",
            title="Build trainer",
            description="Implement train.py",
            status="planning",
            assignee="claude",
            progress=0,
            subtasks=[{"id": "s1", "title": "train loop", "done": False}],
        )
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    fake_dispatcher = _FakeDispatcher()
    fake_sandbox = _FakeSandbox()
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    monkeypatch.setenv("CODEX_SANDBOX_VERIFY_COMMANDS", "pytest -q")
    monkeypatch.setenv("CODEX_SANDBOX_MAX_RETRIES", "1")
    monkeypatch.setenv("CODEX_ENABLE_AUTO_REPAIR", "true")
    monkeypatch.setattr(agent_board_route, "_get_commander", lambda: _FakeCommander())
    monkeypatch.setattr(agent_board_route, "_get_dispatcher", lambda: fake_dispatcher)
    monkeypatch.setattr(agent_board_route, "_get_sandbox", lambda: fake_sandbox)

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/run",
            json={"workspace_dir": str(tmp_path)},
        )

    assert resp.status_code == 200
    assert "sandbox_verify_failed" in resp.text
    assert "repair_attempt_started" in resp.text

    updated_session = agent_board_route._load_session(session.session_id)
    assert updated_session is not None
    task = updated_session.tasks[0]
    assert task.status == "done"
    assert fake_dispatcher.calls == 2  # initial run + one repair run
    assert any("sandbox-verify-attempt-1" in path for path in task.generated_files)


def test_run_stream_sandbox_verify_failure_exhausts_to_human_review(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERBOT_SANDBOX_WORKSPACE", "false")

    class _FakeCommander:
        async def build_codex_prompt(self, task: dict, workspace):
            return f"prompt for {task.get('title')} in {workspace}"

        async def review(self, task: dict, codex_output: str):
            return ReviewResult(approved=True, feedback="Should not reach review")

        def accumulate_wisdom(self, task: dict, output: str):
            return None

    class _FakeDispatcher:
        async def dispatch_auto(self, task_id: str, prompt: str, workspace, **_kwargs):
            return CodexResult(
                task_id=task_id,
                success=True,
                output="generated content",
                files_generated=["src/train.py"],
            )

    class _FakeSandbox:
        executor_type = "FakeSandbox"

        def available(self) -> bool:
            return True

        def run(self, workdir, commands, timeout_sec=300, cache_dir=None, record_meta=True):
            return ExecutionResult(status="failed", exit_code=2, logs="verify always fails")

    session = agent_board_route.BoardSession(
        session_id="board-run-sandbox-exhausted",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir=str(tmp_path),
    )
    session.tasks.append(
        agent_board_route.AgentTask(
            id="task-sandbox-exhausted",
            title="Build trainer",
            description="Implement train.py",
            status="planning",
            assignee="claude",
            progress=0,
            subtasks=[{"id": "s1", "title": "train loop", "done": False}],
        )
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    monkeypatch.setenv("CODEX_SANDBOX_VERIFY_COMMANDS", "pytest -q")
    monkeypatch.setenv("CODEX_SANDBOX_MAX_RETRIES", "0")
    monkeypatch.setenv("CODEX_ENABLE_AUTO_REPAIR", "true")
    monkeypatch.setattr(agent_board_route, "_get_commander", lambda: _FakeCommander())
    monkeypatch.setattr(agent_board_route, "_get_dispatcher", lambda: _FakeDispatcher())
    monkeypatch.setattr(agent_board_route, "_get_sandbox", lambda: _FakeSandbox())

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/run",
            json={"workspace_dir": str(tmp_path)},
        )

    assert resp.status_code == 200
    assert "repair_exhausted" in resp.text

    updated_session = agent_board_route._load_session(session.session_id)
    assert updated_session is not None
    task = updated_session.tasks[0]
    assert task.status == "human_review"
    assert task.codex_output is not None
    assert "Sandbox verification failed" in task.codex_output


def test_run_stream_sandbox_auto_installs_missing_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERBOT_SANDBOX_WORKSPACE", "false")

    class _FakeCommander:
        async def build_codex_prompt(self, task: dict, workspace):
            return f"prompt for {task.get('title')} in {workspace}"

        async def review(self, task: dict, codex_output: str):
            return ReviewResult(approved=True, feedback="Looks correct")

        def accumulate_wisdom(self, task: dict, output: str):
            return None

    class _FakeDispatcher:
        async def dispatch_auto(self, task_id: str, prompt: str, workspace, **_kwargs):
            return CodexResult(
                task_id=task_id,
                success=True,
                output="generated content",
                files_generated=["src/train.py"],
            )

    class _FakeSandbox:
        executor_type = "FakeSandbox"

        def __init__(self):
            self.verify_calls = 0
            self.calls: list[str] = []

        def available(self) -> bool:
            return True

        def run(self, workdir, commands, timeout_sec=300, cache_dir=None, record_meta=True):
            command = commands[0]
            self.calls.append(command)
            if command.startswith("pip install -q -r requirements.txt"):
                return ExecutionResult(status="success", exit_code=0, logs="bootstrap ok")
            if command.startswith("pip install -q statsmodels sqlalchemy"):
                return ExecutionResult(status="success", exit_code=0, logs="deps installed")
            if command == "pytest -q":
                self.verify_calls += 1
                if self.verify_calls == 1:
                    return ExecutionResult(
                        status="failed",
                        exit_code=2,
                        logs=(
                            "ModuleNotFoundError: No module named 'statsmodels'\n"
                            "ModuleNotFoundError: No module named 'sqlalchemy'"
                        ),
                    )
                return ExecutionResult(status="success", exit_code=0, logs="all tests passed")
            return ExecutionResult(status="success", exit_code=0, logs="default pass")

    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    session = agent_board_route.BoardSession(
        session_id="board-run-sandbox-auto-install",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir=str(tmp_path),
    )
    session.tasks.append(
        agent_board_route.AgentTask(
            id="task-sandbox-auto-install",
            title="Build trainer",
            description="Implement train.py",
            status="planning",
            assignee="claude",
            progress=0,
            subtasks=[{"id": "s1", "title": "train loop", "done": False}],
        )
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    fake_sandbox = _FakeSandbox()
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    monkeypatch.setenv("CODEX_SANDBOX_VERIFY_COMMANDS", "pytest -q")
    monkeypatch.setenv("CODEX_SANDBOX_BOOTSTRAP_COMMANDS", "pip install -q -r requirements.txt")
    monkeypatch.setenv("CODEX_SANDBOX_MAX_RETRIES", "0")
    monkeypatch.setattr(agent_board_route, "_get_commander", lambda: _FakeCommander())
    monkeypatch.setattr(agent_board_route, "_get_dispatcher", lambda: _FakeDispatcher())
    monkeypatch.setattr(agent_board_route, "_get_sandbox", lambda: fake_sandbox)

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/run",
            json={"workspace_dir": str(tmp_path)},
        )

    assert resp.status_code == 200
    assert "sandbox_bootstrap_started" in resp.text
    assert "sandbox_dependency_install_finished" in resp.text
    assert fake_sandbox.calls[:4] == [
        "pip install -q -r requirements.txt",
        "pytest -q",
        "pip install -q statsmodels sqlalchemy",
        "pytest -q",
    ]

    updated_session = agent_board_route._load_session(session.session_id)
    assert updated_session is not None
    task = updated_session.tasks[0]
    assert task.status == "done"
    report_paths = [path for path in task.generated_files if "sandbox-verify-attempt-1" in path]
    assert report_paths
    report_text = (tmp_path / report_paths[0]).read_text(encoding="utf-8")
    assert "Dependency Auto-Install" in report_text


def test_repair_prompt_receives_detailed_verify_output(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERBOT_SANDBOX_WORKSPACE", "false")

    class _FakeCommander:
        def __init__(self):
            self.last_verify_details = ""

        async def build_codex_prompt(self, task: dict, workspace):
            return f"prompt for {task.get('title')} in {workspace}"

        async def build_codex_repair_prompt(
            self,
            task: dict,
            workspace,
            *,
            verify_summary: str,
            verify_details: str,
            attempt: int,
        ):
            self.last_verify_details = verify_details
            return (
                f"repair prompt for {task.get('title')} in {workspace} "
                f"(attempt={attempt}, verify={verify_summary})"
            )

        async def review(self, task: dict, codex_output: str):
            return ReviewResult(approved=True, feedback="Looks correct")

        def accumulate_wisdom(self, task: dict, output: str):
            return None

    class _FakeDispatcher:
        def __init__(self):
            self.calls = 0

        async def dispatch_auto(self, task_id: str, prompt: str, workspace, **_kwargs):
            self.calls += 1
            return CodexResult(
                task_id=task_id,
                success=True,
                output=f"generated content call {self.calls}",
                files_generated=["src/train.py"],
            )

    class _FakeSandbox:
        executor_type = "FakeSandbox"

        def __init__(self):
            self._results = [
                ExecutionResult(
                    status="failed",
                    exit_code=1,
                    logs=(
                        "FAILED tests/test_models.py::test_functional_annotation_insert_and_query\n"
                        "sqlalchemy.exc.IntegrityError: NOT NULL constraint failed: proteins.genome_id"
                    ),
                ),
                ExecutionResult(status="success", exit_code=0, logs="verify passed"),
            ]

        def available(self) -> bool:
            return True

        def run(self, workdir, commands, timeout_sec=300, cache_dir=None, record_meta=True):
            if self._results:
                return self._results.pop(0)
            return ExecutionResult(status="success", exit_code=0, logs="default pass")

    session = agent_board_route.BoardSession(
        session_id="board-run-repair-details",
        paper_id="paper-1",
        context_pack_id="cp-1",
        workspace_dir=str(tmp_path),
    )
    session.tasks.append(
        agent_board_route.AgentTask(
            id="task-repair-details",
            title="Fix failing models tests",
            description="Resolve SQLAlchemy test failures",
            status="planning",
            assignee="claude",
            progress=0,
            subtasks=[{"id": "s1", "title": "tests pass", "done": False}],
        )
    )
    agent_board_route._persist_session(session, checkpoint="seed", status="running")

    fake_commander = _FakeCommander()
    fake_dispatcher = _FakeDispatcher()
    monkeypatch.setenv("CODEX_ENABLE_SANDBOX_VERIFY", "true")
    monkeypatch.setenv("CODEX_SANDBOX_VERIFY_COMMANDS", "PYTHONPATH=. pytest -q tests")
    monkeypatch.setenv("CODEX_SANDBOX_MAX_RETRIES", "1")
    monkeypatch.setenv("CODEX_ENABLE_AUTO_REPAIR", "true")
    monkeypatch.setattr(agent_board_route, "_get_commander", lambda: fake_commander)
    monkeypatch.setattr(agent_board_route, "_get_dispatcher", lambda: fake_dispatcher)
    monkeypatch.setattr(agent_board_route, "_get_sandbox", lambda: _FakeSandbox())

    with TestClient(api_main.app) as client:
        resp = client.post(
            f"/api/agent-board/sessions/{session.session_id}/run",
            json={"workspace_dir": str(tmp_path)},
        )

    assert resp.status_code == 200
    assert fake_dispatcher.calls == 2
    assert "IntegrityError" in fake_commander.last_verify_details
    assert "FAILED tests/test_models.py::test_functional_annotation_insert_and_query" in (
        fake_commander.last_verify_details
    )


def test_format_codex_failure_includes_reason_and_diagnostics():
    result = CodexResult(
        task_id="task-format-1",
        success=False,
        error="Agent loop did not finish within 25 iterations.",
        diagnostics={
            "reason_code": "max_iterations_exhausted",
            "steps_executed": 25,
            "effective_max_iterations": 25,
        },
    )

    message, details = agent_board_route._format_codex_failure(result)

    assert "reason=max_iterations_exhausted" in message
    assert details["codex_diagnostics"]["steps_executed"] == 25
    assert details["codex_diagnostics"]["effective_max_iterations"] == 25


def test_format_codex_failure_without_diagnostics_keeps_base_message():
    result = CodexResult(
        task_id="task-format-2",
        success=False,
        error="Codex execution failed.",
    )

    message, details = agent_board_route._format_codex_failure(result)

    assert message == "Codex execution failed."
    assert details == {}
