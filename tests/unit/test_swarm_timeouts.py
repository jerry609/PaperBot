from __future__ import annotations

import asyncio
import sys
import types

import pytest

from paperbot.infrastructure.swarm.claude_commander import ClaudeCommander
from paperbot.infrastructure.swarm.codex_dispatcher import CodexDispatcher


@pytest.mark.asyncio
async def test_codex_dispatch_timeout_returns_failure(monkeypatch, tmp_path):
    class _FakeCompletions:
        async def create(self, **_kwargs):
            await asyncio.sleep(1.2)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

    class _FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    fake_openai = types.SimpleNamespace(AsyncOpenAI=_FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    dispatcher = CodexDispatcher(
        api_key="test-key",
        model="gpt-4o-mini",
        dispatch_timeout_seconds=1,
    )
    result = await dispatcher.dispatch("task-1", "do work", tmp_path)

    assert result.success is False
    assert result.error is not None
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_commander_review_timeout_rejects(monkeypatch):
    class _FakeMessages:
        async def create(self, **_kwargs):
            await asyncio.sleep(1.2)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text='{"approved": false}')]
            )

    class _FakeAnthropicClient:
        def __init__(self, **_kwargs):
            self.messages = _FakeMessages()

    fake_anthropic = types.SimpleNamespace(AsyncAnthropic=_FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    commander = ClaudeCommander(
        api_key="test-key",
        request_timeout_seconds=1,
    )
    review = await commander.review(
        {"title": "Task"},
        codex_output="This is long enough output to trigger review.",
    )

    assert review.approved is False
    assert "timed out" in review.feedback.lower()


@pytest.mark.asyncio
async def test_commander_review_internal_error_rejects(monkeypatch):
    class _FakeMessages:
        async def create(self, **_kwargs):
            raise RuntimeError("anthropic transport error")

    class _FakeAnthropicClient:
        def __init__(self, **_kwargs):
            self.messages = _FakeMessages()

    fake_anthropic = types.SimpleNamespace(AsyncAnthropic=_FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    commander = ClaudeCommander(
        api_key="test-key",
        request_timeout_seconds=1,
    )
    review = await commander.review(
        {"title": "Task"},
        codex_output="This is long enough output to trigger review.",
    )

    assert review.approved is False
    assert "failed" in review.feedback.lower()


@pytest.mark.asyncio
async def test_commander_decompose_timeout_falls_back_to_roadmap(monkeypatch):
    class _FakeMessages:
        async def create(self, **_kwargs):
            await asyncio.sleep(1.2)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="[]")])

    class _FakeAnthropicClient:
        def __init__(self, **_kwargs):
            self.messages = _FakeMessages()

    fake_anthropic = types.SimpleNamespace(AsyncAnthropic=_FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    commander = ClaudeCommander(
        api_key="test-key",
        request_timeout_seconds=1,
    )

    context_pack = {
        "objective": "Reproduce model",
        "task_roadmap": [
            {
                "title": "Implement baseline",
                "description": "Add baseline model code",
                "estimated_difficulty": "medium",
                "acceptance_criteria": ["training starts"],
            }
        ],
        "observations": [],
    }
    tasks = await commander.decompose(context_pack)

    assert len(tasks) == 1
    assert tasks[0]["title"] == "Implement baseline"
