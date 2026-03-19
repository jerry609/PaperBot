"""Unit tests for _emit_codex_event delegation event helper.

TDD: These tests define the required API. They will fail (RED) until
_emit_codex_event is implemented in api/routes/agent_board.py.

asyncio_mode = "strict" — all async tests must have @pytest.mark.asyncio.
"""

from __future__ import annotations

import types
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeEventLog:
    """Fake event log that records append() calls."""

    def __init__(self):
        self.appended: List[Any] = []

    def append(self, envelope) -> None:
        self.appended.append(envelope)


def _make_task(task_id: str = "task-abc", title: str = "Implement model", assignee: str = "codex-a1b2"):
    """Return a minimal AgentTask-like object without importing heavy deps."""
    task = MagicMock()
    task.id = task_id
    task.title = title
    task.assignee = assignee
    return task


def _make_session(session_id: str = "board-test-session"):
    """Return a minimal BoardSession-like object."""
    session = MagicMock()
    session.session_id = session_id
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_codex_event_calls_append(monkeypatch):
    """_emit_codex_event calls event_log.append() with an envelope."""
    from paperbot.application.collaboration.message_schema import EventType
    from paperbot.api.routes.agent_board import _emit_codex_event

    fake_log = _FakeEventLog()
    fake_container = MagicMock()
    fake_container.event_log = fake_log

    # Patch Container inside agent_board module
    import paperbot.api.routes.agent_board as ab_mod

    def _fake_container_instance():
        return fake_container

    monkeypatch.setattr(
        "paperbot.api.routes.agent_board._get_event_log_from_container",
        lambda: fake_log,
    )

    task = _make_task()
    session = _make_session()

    await _emit_codex_event(
        EventType.CODEX_DISPATCHED,
        task,
        session,
        {"assignee": task.assignee},
    )

    assert len(fake_log.appended) == 1, "Expected exactly one append() call"
    envelope = fake_log.appended[0]
    # Payload must contain task_id, task_title, and session_id
    payload = envelope.payload
    assert payload.get("task_id") == "task-abc", f"task_id mismatch: {payload}"
    assert payload.get("task_title") == "Implement model", f"task_title mismatch: {payload}"
    assert payload.get("session_id") == "board-test-session", f"session_id mismatch: {payload}"


@pytest.mark.asyncio
async def test_emit_codex_event_event_type_set(monkeypatch):
    """_emit_codex_event sets envelope.type to the passed event_type."""
    from paperbot.application.collaboration.message_schema import EventType
    from paperbot.api.routes.agent_board import _emit_codex_event

    fake_log = _FakeEventLog()

    monkeypatch.setattr(
        "paperbot.api.routes.agent_board._get_event_log_from_container",
        lambda: fake_log,
    )

    task = _make_task()
    session = _make_session()

    await _emit_codex_event(EventType.CODEX_COMPLETED, task, session, {})

    assert len(fake_log.appended) == 1
    assert fake_log.appended[0].type == EventType.CODEX_COMPLETED


@pytest.mark.asyncio
async def test_emit_codex_event_none_event_log_no_error(monkeypatch):
    """_emit_codex_event silently returns when event_log is None (no exception)."""
    from paperbot.application.collaboration.message_schema import EventType
    from paperbot.api.routes.agent_board import _emit_codex_event

    monkeypatch.setattr(
        "paperbot.api.routes.agent_board._get_event_log_from_container",
        lambda: None,
    )

    task = _make_task()
    session = _make_session()

    # Must not raise
    await _emit_codex_event(EventType.CODEX_DISPATCHED, task, session, {})


@pytest.mark.asyncio
async def test_emit_codex_event_container_raises_no_error(monkeypatch):
    """_emit_codex_event silently returns when _get_event_log_from_container raises."""
    from paperbot.application.collaboration.message_schema import EventType
    from paperbot.api.routes.agent_board import _emit_codex_event

    def _raise():
        raise RuntimeError("No container configured")

    monkeypatch.setattr(
        "paperbot.api.routes.agent_board._get_event_log_from_container",
        _raise,
    )

    task = _make_task()
    session = _make_session()

    # Must not raise
    await _emit_codex_event(EventType.CODEX_DISPATCHED, task, session, {})


@pytest.mark.asyncio
async def test_emit_codex_event_extra_payload_merged(monkeypatch):
    """Extra kwargs passed to _emit_codex_event are merged into the payload."""
    from paperbot.application.collaboration.message_schema import EventType
    from paperbot.api.routes.agent_board import _emit_codex_event

    fake_log = _FakeEventLog()

    monkeypatch.setattr(
        "paperbot.api.routes.agent_board._get_event_log_from_container",
        lambda: fake_log,
    )

    task = _make_task()
    session = _make_session()

    await _emit_codex_event(
        EventType.CODEX_DISPATCHED,
        task,
        session,
        {"assignee": "codex-x9y0", "model": "codex-1"},
    )

    payload = fake_log.appended[0].payload
    assert payload.get("assignee") == "codex-x9y0"
    assert payload.get("model") == "codex-1"
