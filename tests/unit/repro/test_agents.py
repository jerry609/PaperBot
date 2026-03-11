"""Unit tests for repro agents."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from repro.agents.base_agent import AgentResult, AgentStatus, BaseAgent
from repro.agents.verification_agent import VerificationAgent, VerificationReport


class ConcreteAgent(BaseAgent):
    def __init__(self, return_value=None, should_fail=False, **kwargs):
        super().__init__(name="TestAgent", **kwargs)
        self.return_value = return_value
        self.should_fail = should_fail
        self.execute_count = 0

    async def execute(self, context):
        self.execute_count += 1
        if self.should_fail:
            raise ValueError("Test error")
        return AgentResult.success(data=self.return_value)


def test_agent_status_values():
    assert AgentStatus.IDLE.value == "idle"
    assert AgentStatus.RUNNING.value == "running"
    assert AgentStatus.COMPLETED.value == "completed"
    assert AgentStatus.FAILED.value == "failed"
    assert AgentStatus.WAITING.value == "waiting"


def test_agent_result_success():
    result = AgentResult.success(data={"key": "value"}, metadata={"step": 1})

    assert result.status is AgentStatus.COMPLETED
    assert result.data == {"key": "value"}
    assert result.error is None
    assert result.metadata == {"step": 1}


def test_agent_result_failure():
    result = AgentResult.failure("Something went wrong")

    assert result.status is AgentStatus.FAILED
    assert result.data is None
    assert result.error == "Something went wrong"


def test_agent_result_to_dict():
    result = AgentResult(
        status=AgentStatus.COMPLETED,
        data={"result": 42},
        error=None,
        duration_seconds=1.5,
        messages=["Step 1 done", "Step 2 done"],
        metadata={"info": "test"},
    )

    payload = result.to_dict()

    assert payload["status"] == "completed"
    assert "42" in payload["data"]
    assert payload["error"] is None
    assert payload["duration_seconds"] == 1.5
    assert len(payload["messages"]) == 2
    assert payload["metadata"]["info"] == "test"


@pytest.mark.asyncio
async def test_base_agent_successful_run():
    agent = ConcreteAgent(return_value={"answer": 42})

    result = await agent.run({})

    assert result.status is AgentStatus.COMPLETED
    assert result.data == {"answer": 42}
    assert agent.execute_count == 1


@pytest.mark.asyncio
async def test_base_agent_failed_run():
    agent = ConcreteAgent(should_fail=True)

    result = await agent.run({})

    assert result.status is AgentStatus.FAILED
    assert "Test error" in result.error


@pytest.mark.asyncio
async def test_base_agent_status_tracking():
    agent = ConcreteAgent()

    assert agent.status is AgentStatus.IDLE
    await agent.run({})
    assert agent.status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_base_agent_duration_tracking():
    agent = ConcreteAgent()

    result = await agent.run({})

    assert result.duration_seconds >= 0


def test_base_agent_log_method():
    agent = ConcreteAgent()

    agent.log("Test message")

    assert len(agent._messages) == 1
    assert "Test message" in agent._messages[0]


def test_verification_report_properties():
    partial = VerificationReport(syntax_ok=True, imports_ok=True, tests_ok=False, smoke_ok=False)
    failed = VerificationReport(syntax_ok=True, imports_ok=False)
    complete = VerificationReport(syntax_ok=True, imports_ok=True, tests_ok=True, smoke_ok=True)

    assert partial.all_passed is True
    assert partial.fully_passed is False
    assert failed.all_passed is False
    assert complete.fully_passed is True


def test_verification_report_to_dict():
    report = VerificationReport(
        syntax_ok=True,
        imports_ok=True,
        errors=["Error 1"],
        warnings=["Warning 1"],
    )

    payload = report.to_dict()

    assert payload["syntax_ok"] is True
    assert payload["imports_ok"] is True
    assert payload["all_passed"] is True
    assert payload["fully_passed"] is False
    assert payload["errors"] == ["Error 1"]
    assert payload["warnings"] == ["Warning 1"]


@pytest.fixture
def verification_agent():
    return VerificationAgent(timeout=10, run_tests=False, run_smoke=False)


@pytest.mark.asyncio
async def test_verification_agent_missing_output_dir(verification_agent):
    result = await verification_agent.run({})

    assert result.status is AgentStatus.FAILED
    assert "output_dir" in result.error


@pytest.mark.asyncio
async def test_verification_agent_nonexistent_output_dir(verification_agent):
    result = await verification_agent.run({"output_dir": "/nonexistent/path"})

    assert result.status is AgentStatus.FAILED
    assert "does not exist" in result.error


@pytest.mark.asyncio
async def test_verification_agent_syntax_check_valid(verification_agent):
    with TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").write_text("def hello(): pass", encoding="utf-8")

        result = await verification_agent.run({"output_dir": tmpdir})

    assert result.status is AgentStatus.COMPLETED
    assert result.data["report"].syntax_ok is True


@pytest.mark.asyncio
async def test_verification_agent_syntax_check_invalid(verification_agent):
    with TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "broken.py").write_text("def broken(", encoding="utf-8")

        result = await verification_agent.run({"output_dir": tmpdir})

    assert result.status is AgentStatus.COMPLETED
    assert result.data["report"].syntax_ok is False
    assert result.data["report"].errors


@pytest.mark.asyncio
async def test_verification_agent_import_check(verification_agent):
    with TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "simple.py").write_text("x = 1", encoding="utf-8")

        result = await verification_agent.run({"output_dir": tmpdir})

    assert result.status is AgentStatus.COMPLETED
    assert result.data["report"].imports_ok is True


@pytest.mark.asyncio
async def test_verification_agent_updates_context_with_report(verification_agent):
    with TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").write_text("x = 1", encoding="utf-8")
        context = {"output_dir": tmpdir}

        await verification_agent.run(context)

    assert "verification_report" in context
    assert isinstance(context["verification_report"], VerificationReport)
