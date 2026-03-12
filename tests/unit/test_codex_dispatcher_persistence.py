from __future__ import annotations

import sys
import types

import pytest

from paperbot.infrastructure.swarm.codex_dispatcher import CodexDispatcher


@pytest.mark.asyncio
async def test_dispatch_persists_generated_files(monkeypatch, tmp_path):
    output = (
        "File: src/train.py\n"
        "```python\n"
        "print('train')\n"
        "```\n\n"
        "```ts filename=web/src/app.ts\n"
        "export const ok = true\n"
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
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch("task-123", "prompt", tmp_path)

    assert result.success is True
    assert "src/train.py" in result.files_generated
    assert "web/src/app.ts" in result.files_generated
    assert "reviews/task-123-user-review.md" in result.files_generated
    assert (tmp_path / "src/train.py").read_text(encoding="utf-8") == "print('train')\n"
    assert (tmp_path / "web/src/app.ts").read_text(encoding="utf-8") == "export const ok = true\n"
    review_doc = (tmp_path / "reviews/task-123-user-review.md").read_text(encoding="utf-8")
    assert "## What Was Added" in review_doc
    assert "## Why This Approach" in review_doc
    assert "## File & Function Overview" in review_doc
    assert "src/train.py" in review_doc


@pytest.mark.asyncio
async def test_dispatch_writes_fallback_when_no_file_blocks(monkeypatch, tmp_path):
    output = "No code fences were returned."

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
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch("task-xyz", "prompt", tmp_path)

    assert result.success is True
    assert "task-xyz-output.md" in result.files_generated
    assert "reviews/task-xyz-user-review.md" in result.files_generated
    assert (tmp_path / "task-xyz-output.md").read_text(encoding="utf-8") == output
    review_doc = (tmp_path / "reviews/task-xyz-user-review.md").read_text(encoding="utf-8")
    assert "No explicit rationale provided" in review_doc
