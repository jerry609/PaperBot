"""Integration coverage for the current DeepCode-style repro stack."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from repro import (
    AgentResult,
    Blueprint,
    CodeKnowledgeBase,
    CodeMemory,
    EnvironmentSpec,
    GenerationNode,
    ImplementationSpec,
    NodeResult,
    Orchestrator,
    OrchestratorConfig,
    PaperContext,
    ReproAgent,
    ReproductionPlan,
    ReproPhase,
)


@pytest.fixture
def paper_context():
    return PaperContext(
        title="Attention Is All You Need",
        abstract=(
            "We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms."
        ),
        method_section=(
            "The Transformer follows an encoder-decoder structure using stacked "
            "self-attention and point-wise fully connected layers."
        ),
        hyperparameters={"d_model": 512, "n_heads": 8, "n_layers": 6},
    )


def test_memory_and_rag_integration():
    memory = CodeMemory(max_context_tokens=2000)
    knowledge_base = CodeKnowledgeBase.from_builtin()

    memory.add_file(
        "config.py",
        "class Config:\n    d_model = 512\n    n_heads = 8\n",
        purpose="Configuration",
    )
    patterns = knowledge_base.search("transformer attention", k=2)
    context = memory.get_relevant_context("model.py", "Transformer model")

    assert patterns
    assert "config.py" in context


def test_generation_node_initializes_memory_and_knowledge_base():
    node = GenerationNode(max_context_tokens=2000, use_rag=True)

    assert node.memory is not None
    assert node.knowledge_base is not None
    assert node.memory.max_context_tokens == 2000


def test_repro_agent_mode_switching():
    legacy_agent = ReproAgent({"use_orchestrator": False})
    orchestrated_agent = ReproAgent({"use_orchestrator": True})

    assert legacy_agent.use_orchestrator is False
    assert orchestrated_agent.use_orchestrator is True


@pytest.mark.asyncio
async def test_orchestrator_mock_pipeline_completes(tmp_path, paper_context):
    orchestrator = Orchestrator(config=OrchestratorConfig(max_repair_loops=1, output_dir=tmp_path))
    mock_plan = MagicMock()
    mock_plan.file_structure = {"main.py": "entry", "model.py": "model"}

    async def planning_run(context):
        context["plan"] = mock_plan
        context["blueprint"] = Blueprint(architecture_type="transformer", domain="nlp")
        return AgentResult.success(data={"plan": mock_plan})

    async def coding_run(context):
        context["generated_files"] = {
            "main.py": "print('hello')",
            "model.py": "class Model: pass",
        }
        for name, content in context["generated_files"].items():
            (tmp_path / name).write_text(content)
        return AgentResult.success(data={"generated_files": context["generated_files"]})

    report = SimpleNamespace(
        all_passed=True,
        to_dict=lambda: {"syntax_ok": True, "imports_ok": True},
    )

    async def verification_run(context):
        context["verification_report"] = report
        return AgentResult.success(data={"report": report})

    orchestrator.planning_agent.run = planning_run
    orchestrator.coding_agent.run = coding_run
    orchestrator.verification_agent.run = verification_run

    result = await orchestrator.run(paper_context)

    assert result.status is ReproPhase.COMPLETED
    assert "planning" in result.phases_completed
    assert "generation" in result.phases_completed
    assert (tmp_path / "main.py").read_text() == "print('hello')"


@pytest.mark.asyncio
async def test_repro_agent_legacy_mode_runs_with_mocked_nodes(paper_context):
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        agent = ReproAgent({"use_orchestrator": False, "timeout_sec": 10})
        agent.planning_node.run = AsyncMock(
            return_value=NodeResult.ok(
                ReproductionPlan(
                    project_name="Attention Is All You Need",
                    description="Transformer reproduction",
                    file_structure={"main.py": "entry"},
                    dependencies=[],
                )
            )
        )
        agent.environment_node.run = AsyncMock(
            return_value=NodeResult.ok(
                EnvironmentSpec(
                    python_version="3.10",
                    base_image="python:3.10-slim",
                    pip_requirements=[],
                )
            )
        )
        agent.analysis_node.run = AsyncMock(
            return_value=NodeResult.ok(
                ImplementationSpec(
                    model_type="mlp",
                    layers=[],
                    optimizer="adam",
                    extra_params={},
                )
            )
        )
        agent.generation_node.run = AsyncMock(return_value=NodeResult.ok({"main.py": "print('hello')"}))
        agent.verification_node.run = AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                data=SimpleNamespace(
                    all_passed=True,
                    syntax_ok=True,
                    imports_ok=True,
                    errors=[],
                    repairs_attempted=0,
                    repairs_successful=0,
                    to_dict=lambda: {"all_passed": True},
                ),
            )
        )

        result = await agent.reproduce_from_paper(paper_context, output_dir=output_dir)

    assert result.status is ReproPhase.COMPLETED
    assert "planning" in result.phases_completed
    assert "generation" in result.phases_completed


def test_repro_agent_orchestrator_mode_init(tmp_path):
    agent = ReproAgent(
        {
            "use_orchestrator": True,
            "use_rag": True,
            "max_context_tokens": 4000,
        }
    )

    orchestrator = agent.get_orchestrator(tmp_path)

    assert orchestrator is not None
    assert orchestrator.config.use_rag is True
