"""Unit tests for the repro orchestrator."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from repro.agents.base_agent import AgentResult, AgentStatus
from repro.models import PaperContext, ReproPhase
from repro.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    ParallelOrchestrator,
    PipelineProgress,
    PipelineStage,
)


def test_orchestrator_config_defaults():
    config = OrchestratorConfig()

    assert config.max_repair_loops == 3
    assert config.parallel_agents is True
    assert config.timeout_seconds == 300
    assert config.output_dir is None
    assert config.use_rag is True
    assert config.max_context_tokens == 8000


def test_orchestrator_config_custom_values():
    config = OrchestratorConfig(
        max_repair_loops=5,
        parallel_agents=False,
        output_dir=Path("/tmp/test"),
        use_rag=False,
    )

    assert config.max_repair_loops == 5
    assert config.parallel_agents is False
    assert config.output_dir == Path("/tmp/test")
    assert config.use_rag is False


def test_pipeline_stage_values():
    assert PipelineStage.PLANNING.value == "planning"
    assert PipelineStage.CODING.value == "coding"
    assert PipelineStage.VERIFICATION.value == "verification"
    assert PipelineStage.DEBUGGING.value == "debugging"
    assert PipelineStage.COMPLETED.value == "completed"
    assert PipelineStage.FAILED.value == "failed"


def test_pipeline_progress_defaults():
    progress = PipelineProgress()

    assert progress.current_stage is PipelineStage.PLANNING
    assert progress.stages_completed == []
    assert progress.repair_loop_count == 0
    assert progress.agent_results == {}
    assert progress.duration_seconds == 0.0


def test_pipeline_progress_duration_and_dict():
    progress = PipelineProgress(
        current_stage=PipelineStage.CODING,
        stages_completed=["planning"],
        repair_loop_count=1,
    )
    progress.start_time = datetime.now()
    progress.end_time = datetime.now()

    payload = progress.to_dict()

    assert payload["current_stage"] == "coding"
    assert payload["stages_completed"] == ["planning"]
    assert payload["repair_loop_count"] == 1
    assert "duration_seconds" in payload


@pytest.fixture
def paper_context():
    return PaperContext(
        title="Test Paper",
        abstract="This is a test abstract for testing purposes.",
    )


@pytest.fixture
def orchestrator():
    return Orchestrator(config=OrchestratorConfig(max_repair_loops=1))


def test_orchestrator_initialization(orchestrator):
    assert orchestrator.planning_agent is not None
    assert orchestrator.coding_agent is not None
    assert orchestrator.verification_agent is not None
    assert orchestrator.debugging_agent is not None
    assert orchestrator.context == {}


@pytest.mark.asyncio
async def test_orchestrator_progress_callback(paper_context):
    progress_updates = []

    def on_progress(progress):
        progress_updates.append(progress.current_stage)

    orchestrator = Orchestrator(config=OrchestratorConfig(max_repair_loops=1), on_progress=on_progress)
    orchestrator.planning_agent.run = AsyncMock(return_value=AgentResult.failure("Skip"))

    await orchestrator.run(paper_context)

    assert progress_updates


@pytest.mark.asyncio
async def test_orchestrator_run_planning_failure(orchestrator, paper_context):
    orchestrator.planning_agent.run = AsyncMock(return_value=AgentResult.failure("Planning failed"))

    result = await orchestrator.run(paper_context)

    assert result.status is ReproPhase.FAILED
    assert "Planning failed" in result.error


@pytest.mark.asyncio
async def test_orchestrator_run_coding_failure(orchestrator, paper_context):
    orchestrator.planning_agent.run = AsyncMock(return_value=AgentResult.success(data={"plan": {}}))
    orchestrator.context["plan"] = {"files": []}
    orchestrator.coding_agent.run = AsyncMock(return_value=AgentResult.failure("Coding failed"))

    result = await orchestrator.run(paper_context)

    assert result.status is ReproPhase.FAILED
    assert "Coding failed" in result.error


@pytest.mark.asyncio
async def test_orchestrator_context_shared_between_agents(orchestrator, paper_context):
    async def planning_run(context):
        context["plan"] = {"files": ["main.py"]}
        return AgentResult.success(data={"plan": context["plan"]})

    async def coding_run(context):
        assert "plan" in context
        context["generated_files"] = {"main.py": "print('hello')"}
        return AgentResult.success(data={"generated_files": context["generated_files"]})

    report = MagicMock()
    report.all_passed = True
    report.to_dict.return_value = {}

    async def verification_run(context):
        context["verification_report"] = report
        return AgentResult.success(data={"report": report})

    orchestrator.planning_agent.run = planning_run
    orchestrator.coding_agent.run = coding_run
    orchestrator.verification_agent.run = verification_run

    result = await orchestrator.run(paper_context)

    assert "plan" in orchestrator.context
    assert result.status is ReproPhase.COMPLETED


@pytest.mark.asyncio
async def test_orchestrator_result_includes_timing(orchestrator, paper_context):
    orchestrator.planning_agent.run = AsyncMock(return_value=AgentResult.success(data={}))
    orchestrator.coding_agent.run = AsyncMock(return_value=AgentResult.success(data={"generated_files": {}}))

    report = MagicMock()
    report.all_passed = True
    report.to_dict.return_value = {}

    async def verify_run(context):
        context["verification_report"] = report
        return AgentResult.success(data={"report": report})

    orchestrator.verification_agent.run = verify_run

    result = await orchestrator.run(paper_context)

    assert result.total_duration_sec is not None
    assert result.total_duration_sec >= 0


@pytest.mark.asyncio
async def test_orchestrator_repair_loop_count(orchestrator, paper_context):
    orchestrator.planning_agent.run = AsyncMock(return_value=AgentResult.success(data={}))
    orchestrator.coding_agent.run = AsyncMock(return_value=AgentResult.success(data={"generated_files": {}}))

    report = MagicMock()
    report.all_passed = False
    report.to_dict.return_value = {}

    async def verify_run(context):
        context["verification_report"] = report
        context["error"] = "Verification failed"
        return AgentResult.success(data={"report": report})

    orchestrator.verification_agent.run = verify_run
    orchestrator.debugging_agent.run = AsyncMock(return_value=AgentResult.success(data={}))

    result = await orchestrator.run(paper_context)

    assert result.retry_count > 0


@pytest.mark.asyncio
async def test_orchestrator_timeout_is_enforced(paper_context):
    orchestrator = Orchestrator(config=OrchestratorConfig(timeout_seconds=0.01))

    async def slow_planning(_context):
        await asyncio.sleep(0.05)
        return AgentResult.success(data={})

    orchestrator.planning_agent.run = slow_planning

    result = await orchestrator.run(paper_context)

    assert result.status is ReproPhase.FAILED
    assert result.error == "Pipeline timed out after 0.01s"
    assert orchestrator.progress.current_stage is PipelineStage.FAILED


def test_parallel_orchestrator_inherits_from_orchestrator():
    parallel = ParallelOrchestrator()

    assert isinstance(parallel, Orchestrator)
    assert hasattr(parallel, "run_parallel")
    assert callable(parallel.run_parallel)
