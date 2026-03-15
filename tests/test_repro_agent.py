"""Tests for the public repro agent APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from paperbot.repro import (
    EnvironmentSpec,
    ExecutionResult,
    ImplementationSpec,
    NodeResult,
    Orchestrator,
    PaperContext,
    ReproAgent,
    ReproductionPlan,
    ReproductionResult,
    ReproPhase,
)


class FakeExecutor:
    def __init__(self, result: ExecutionResult | None = None):
        self.result = result or ExecutionResult(status="success", exit_code=0, logs="ok")
        self.calls = []

    def available(self) -> bool:
        return True

    def run(
        self,
        workdir: Path,
        commands,
        timeout_sec: int = 300,
        cache_dir=None,
        record_meta: bool = True,
    ):
        self.calls.append((workdir, commands, timeout_sec))
        return self.result


def _verification_report(
    *,
    all_passed: bool,
    syntax_ok: bool,
    imports_ok: bool,
    repairs_attempted: int = 0,
):
    return SimpleNamespace(
        all_passed=all_passed,
        syntax_ok=syntax_ok,
        imports_ok=imports_ok,
        errors=[] if all_passed else ["verification failed"],
        repairs_attempted=repairs_attempted,
        repairs_successful=repairs_attempted if all_passed else 0,
        to_dict=lambda: {
            "all_passed": all_passed,
            "syntax_ok": syntax_ok,
            "imports_ok": imports_ok,
        },
    )


@pytest.mark.asyncio
async def test_legacy_run_executes_generated_plan():
    agent = ReproAgent({})
    agent.executor = FakeExecutor()
    agent.generate_plan = AsyncMock(return_value={"commands": ["echo hello"], "repo_path": "."})

    result = await agent.run(Path("."))

    assert result["passed"] is True
    assert result["results"][0]["commands"] == ["echo hello"]
    assert result["score"] == 1.0


@pytest.mark.asyncio
async def test_reproduce_from_paper_legacy_pipeline_writes_files(tmp_path):
    agent = ReproAgent({"use_orchestrator": False})
    agent.planning_node.run = AsyncMock(
        return_value=NodeResult.ok(
            ReproductionPlan(
                project_name="Test",
                description="desc",
                file_structure={"main.py": "entry"},
                dependencies=["numpy"],
            )
        )
    )
    agent.environment_node.run = AsyncMock(
        return_value=NodeResult.ok(
            EnvironmentSpec(base_image="python:3.10-slim", pip_requirements=["numpy"])
        )
    )
    agent.analysis_node.run = AsyncMock(
        return_value=NodeResult.ok(ImplementationSpec(model_type="mlp", extra_params={}))
    )
    agent.generation_node.run = AsyncMock(return_value=NodeResult.ok({"main.py": "print('hello')"}))
    agent.verification_node.run = AsyncMock(
        return_value=NodeResult.ok(
            _verification_report(all_passed=True, syntax_ok=True, imports_ok=True)
        )
    )

    result = await agent.reproduce_from_paper(
        PaperContext(title="Test", abstract="Abs", method_section="Method"),
        output_dir=tmp_path,
    )

    assert result.status is ReproPhase.COMPLETED
    assert result.generated_files == {"main.py": "print('hello')"}
    assert (tmp_path / "main.py").read_text() == "print('hello')"
    assert "planning" in result.phases_completed
    assert "generation" in result.phases_completed
    assert "verification" in result.phases_completed


@pytest.mark.asyncio
async def test_reproduce_from_paper_uses_orchestrator_mode_when_enabled(tmp_path):
    agent = ReproAgent({"use_orchestrator": True})
    expected = ReproductionResult(
        paper_title="Test",
        status=ReproPhase.COMPLETED,
        phases_completed=["planning", "generation", "verification"],
    )
    agent._reproduce_with_orchestrator = AsyncMock(return_value=expected)

    result = await agent.reproduce_from_paper(
        PaperContext(title="Test", abstract="Abs"),
        output_dir=tmp_path,
    )

    assert result is expected
    agent._reproduce_with_orchestrator.assert_awaited_once()


def test_get_orchestrator_propagates_timeout_config(tmp_path):
    agent = ReproAgent({"use_orchestrator": True, "timeout_sec": 42})

    orchestrator = agent.get_orchestrator(tmp_path)

    assert isinstance(orchestrator, Orchestrator)
    assert orchestrator.config.timeout_seconds == 42
