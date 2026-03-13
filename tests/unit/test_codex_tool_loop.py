from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

from paperbot.infrastructure.swarm.codex_dispatcher import CodexDispatcher
from paperbot.repro.base_executor import BaseExecutor
from paperbot.repro.execution_result import ExecutionResult


class _FakeToolCall:
    def __init__(self, name: str, args: Any, call_id: str = "call-1"):
        self.id = call_id
        if isinstance(args, str):
            arguments = args
        else:
            arguments = json.dumps(args)
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class _FakeMessage:
    def __init__(
        self, *, content: str | None = None, tool_calls: list[_FakeToolCall] | None = None
    ):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self):
        payload: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return payload


class _FakeChoice:
    def __init__(self, finish_reason: str, message: _FakeMessage):
        self.finish_reason = finish_reason
        self.message = message


class _FakeResponse:
    def __init__(self, choice: _FakeChoice):
        self.choices = [choice]


def _install_fake_openai(monkeypatch, responses: list[_FakeResponse]):
    class _FakeCompletions:
        def __init__(self):
            self._responses = list(responses)

        async def create(self, **_kwargs):
            if not self._responses:
                raise RuntimeError("No fake responses left")
            return self._responses.pop(0)

    class _FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_FakeOpenAIClient))


@pytest.mark.asyncio
async def test_tool_loop_write_and_done(monkeypatch, tmp_path):
    responses = [
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(
                    tool_calls=[
                        _FakeToolCall(
                            "write_file", {"path": "src/hello.py", "content": "print(1)\n"}
                        )
                    ]
                ),
            )
        ),
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(tool_calls=[_FakeToolCall("task_done", {"summary": "finished"})]),
            )
        ),
    ]
    _install_fake_openai(monkeypatch, responses)

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch_with_tools("task-1", "do work", tmp_path)

    assert result.success is True
    assert result.output == "finished"
    assert "src/hello.py" in result.files_generated
    assert (tmp_path / "src/hello.py").read_text(encoding="utf-8") == "print(1)\n"


@pytest.mark.asyncio
async def test_tool_loop_text_fallback(monkeypatch, tmp_path):
    output = "File: src/fallback.py\n```python\nx = 1\n```\n"
    responses = [_FakeResponse(_FakeChoice("stop", _FakeMessage(content=output)))]
    _install_fake_openai(monkeypatch, responses)

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch_with_tools("task-2", "do work", tmp_path)

    assert result.success is True
    assert "src/fallback.py" in result.files_generated
    assert "reviews/task-2-user-review.md" in result.files_generated
    assert (tmp_path / "src/fallback.py").exists()


@pytest.mark.asyncio
async def test_tool_loop_max_iterations(monkeypatch, tmp_path):
    responses = [
        _FakeResponse(
            _FakeChoice(
                "tool_calls", _FakeMessage(tool_calls=[_FakeToolCall("list_files", {"path": "."})])
            )
        ),
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(tool_calls=[_FakeToolCall("list_files", {"path": "src"})]),
            )
        ),
    ]
    _install_fake_openai(monkeypatch, responses)

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch_with_tools("task-3", "do work", tmp_path, max_iterations=2)

    assert result.success is False
    assert result.error is not None
    assert "within 2 iterations" in result.error
    assert result.diagnostics.get("reason_code") == "max_iterations_exhausted"


@pytest.mark.asyncio
async def test_tool_loop_timeout(monkeypatch, tmp_path):
    class _SlowCompletions:
        async def create(self, **_kwargs):
            await asyncio.sleep(1.2)
            return _FakeResponse(_FakeChoice("stop", _FakeMessage(content="ok")))

    class _SlowOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=_SlowCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_SlowOpenAIClient))

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=1
    )
    result = await dispatcher.dispatch_with_tools("task-4", "do work", tmp_path)

    assert result.success is False
    assert result.error is not None
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_tool_loop_stagnation_early_stop(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_LOOP_STAGNATION_STEPS", "2")
    monkeypatch.setenv("CODEX_LOOP_AUTO_BUDGET", "false")

    responses = [
        _FakeResponse(
            _FakeChoice(
                "tool_calls", _FakeMessage(tool_calls=[_FakeToolCall("list_files", {"path": "."})])
            )
        ),
        _FakeResponse(
            _FakeChoice(
                "tool_calls", _FakeMessage(tool_calls=[_FakeToolCall("list_files", {"path": "."})])
            )
        ),
        _FakeResponse(
            _FakeChoice(
                "tool_calls", _FakeMessage(tool_calls=[_FakeToolCall("list_files", {"path": "."})])
            )
        ),
    ]
    _install_fake_openai(monkeypatch, responses)

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch_with_tools("task-stagnation", "do work", tmp_path, max_iterations=10)

    assert result.success is False
    assert result.error is not None
    assert "stagnation" in result.error.lower()
    assert result.diagnostics.get("reason_code") == "stagnation_detected"


@pytest.mark.asyncio
async def test_tool_loop_auto_budget_extension(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_LOOP_AUTO_BUDGET", "true")
    monkeypatch.setenv("CODEX_LOOP_STAGNATION_STEPS", "10")
    monkeypatch.setenv("CODEX_LOOP_AUTO_EXTEND_STEPS", "1")
    monkeypatch.setenv("CODEX_LOOP_AUTO_EXTEND_MAX", "1")
    monkeypatch.setenv("CODEX_LOOP_NEAR_LIMIT_WINDOW", "1")

    responses = [
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(tool_calls=[_FakeToolCall("write_file", {"path": "src/a.py", "content": "a=1\n"})]),
            )
        ),
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(tool_calls=[_FakeToolCall("write_file", {"path": "src/b.py", "content": "b=2\n"})]),
            )
        ),
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(tool_calls=[_FakeToolCall("task_done", {"summary": "finished with extension"})]),
            )
        ),
    ]
    _install_fake_openai(monkeypatch, responses)

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch_with_tools(
        "task-auto-budget",
        "do work",
        tmp_path,
        max_iterations=2,
    )

    assert result.success is True
    assert result.output == "finished with extension"
    assert result.diagnostics.get("extensions_used") == 1
    assert result.diagnostics.get("effective_max_iterations") == 3


@pytest.mark.asyncio
async def test_tool_loop_on_step_called(monkeypatch, tmp_path):
    responses = [
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(
                    tool_calls=[
                        _FakeToolCall("write_file", {"path": "src/a.py", "content": "a = 1\n"})
                    ]
                ),
            )
        ),
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(tool_calls=[_FakeToolCall("task_done", {"summary": "done"})]),
            )
        ),
    ]
    _install_fake_openai(monkeypatch, responses)

    steps: list[tuple[int, str, dict[str, Any], str]] = []

    async def _on_step(step: int, tool_name: str, args: dict[str, Any], observation: str):
        steps.append((step, tool_name, args, observation))

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch_with_tools("task-5", "do work", tmp_path, on_step=_on_step)

    assert result.success is True
    assert len(steps) == 2
    assert steps[0][1] == "write_file"
    assert steps[1][1] == "task_done"


@pytest.mark.asyncio
async def test_tool_loop_handles_malformed_tool_arguments(monkeypatch, tmp_path):
    responses = [
        _FakeResponse(
            _FakeChoice(
                "tool_calls",
                _FakeMessage(
                    tool_calls=[_FakeToolCall("write_file", "{bad-json", call_id="call-x")]
                ),
            )
        ),
        _FakeResponse(_FakeChoice("stop", _FakeMessage(content="No file output"))),
    ]
    _install_fake_openai(monkeypatch, responses)

    observations: list[str] = []

    async def _on_step(_step: int, _tool_name: str, _args: dict[str, Any], observation: str):
        observations.append(observation)

    dispatcher = CodexDispatcher(
        api_key="test-key", model="gpt-4o-mini", dispatch_timeout_seconds=5
    )
    result = await dispatcher.dispatch_with_tools("task-6", "do work", tmp_path, on_step=_on_step)

    assert result.success is True
    assert any("malformed tool arguments" in text for text in observations)


class _AvailableSandbox(BaseExecutor):
    def available(self) -> bool:
        return True

    def run(self, workdir, commands, timeout_sec=300, cache_dir=None, record_meta=True):
        return ExecutionResult(status="success", exit_code=0, logs="ok")


def test_tool_system_prompt_includes_sandbox_install_guidance():
    dispatcher = CodexDispatcher(api_key="test-key")
    prompt = dispatcher._tool_system_prompt(sandbox=_AvailableSandbox())

    assert "Pre-installed packages include" in prompt
    assert "pip install -q" in prompt


def test_tool_system_prompt_without_sandbox_omits_install_guidance():
    dispatcher = CodexDispatcher(api_key="test-key")
    prompt = dispatcher._tool_system_prompt(sandbox=None)

    assert "Pre-installed packages include" not in prompt
