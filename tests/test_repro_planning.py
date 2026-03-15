"""Compatibility tests for the current repro planning layer."""

from __future__ import annotations

import pytest

from paperbot.repro import Blueprint, PaperContext, PlanningNode, ReproductionPlan


@pytest.fixture
def paper_context():
    return PaperContext(
        title="Test Paper",
        abstract="Abstract",
        method_section="Method",
    )


def test_fallback_plan_without_blueprint_uses_default_files(paper_context):
    node = PlanningNode()

    plan = node._fallback_plan(paper_context)

    assert isinstance(plan, ReproductionPlan)
    assert plan.entry_point == "main.py"
    assert "main.py" in plan.file_structure
    assert "torch" in plan.dependencies


def test_fallback_plan_with_blueprint_infers_structure_and_dependencies(paper_context):
    node = PlanningNode()
    blueprint = Blueprint(
        architecture_type="transformer",
        module_hierarchy={"model": ["encoder", "decoder"], "trainer": []},
        optimization_strategy="Adam with warmup",
        loss_functions=["cross_entropy"],
        input_output_spec={"input": "tokens"},
        framework_hints=["pytorch", "transformers"],
    )

    plan = node._fallback_plan(paper_context, blueprint)

    assert "model.py" in plan.file_structure
    assert "trainer.py" in plan.file_structure
    assert "losses.py" in plan.file_structure
    assert "torch" in plan.dependencies
    assert "transformers" in plan.dependencies


def test_parse_plan_json_enriches_dependencies_from_blueprint_hints(paper_context):
    node = PlanningNode()
    blueprint = Blueprint(framework_hints=["pytorch", "transformers"])
    response = """
    {
        "files": [{"path": "main.py", "purpose": "Entry point"}],
        "components": ["Model"],
        "dependencies": ["numpy"]
    }
    """

    plan = node._parse_plan(response, paper_context, blueprint)

    assert plan.project_name == "Test Paper"
    assert plan.file_structure == {"main.py": "Entry point"}
    assert "numpy" in plan.dependencies
    assert "torch" in plan.dependencies
    assert "transformers" in plan.dependencies


def test_parse_plan_invalid_json_returns_fallback(paper_context):
    node = PlanningNode()

    plan = node._parse_plan("invalid json", paper_context)

    assert plan.description == "Abstract"


@pytest.mark.asyncio
async def test_planning_node_run_uses_fallback_when_no_llm_is_available(paper_context):
    node = PlanningNode()

    result = await node.run(paper_context)

    assert result.success is True
    assert isinstance(result.data, ReproductionPlan)
    assert "main.py" in result.data.file_structure
