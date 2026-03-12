"""End-to-end style tests for the legacy repro pipeline control flow."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from repro import (
    EnvironmentSpec,
    ImplementationSpec,
    NodeResult,
    PaperContext,
    ReproAgent,
    ReproductionPlan,
    ReproPhase,
)


def _legacy_agent_with_common_nodes():
    agent = ReproAgent({"use_orchestrator": False})
    agent.planning_node.run = AsyncMock(
        return_value=NodeResult.ok(
            ReproductionPlan(
                project_name="E2E Paper",
                description="Abstract",
                file_structure={"main.py": "entry"},
                dependencies=[],
            )
        )
    )
    agent.environment_node.run = AsyncMock(
        return_value=NodeResult.ok(
            EnvironmentSpec(base_image="python:3.10-slim", pip_requirements=[])
        )
    )
    agent.analysis_node.run = AsyncMock(
        return_value=NodeResult.ok(ImplementationSpec(model_type="mlp", extra_params={}))
    )
    agent.generation_node.run = AsyncMock(
        return_value=NodeResult.ok({"main.py": "print('generated')"})
    )
    return agent


def _report(
    *,
    all_passed: bool,
    syntax_ok: bool,
    imports_ok: bool,
    errors=None,
    repairs_attempted: int = 0,
):
    return SimpleNamespace(
        all_passed=all_passed,
        syntax_ok=syntax_ok,
        imports_ok=imports_ok,
        errors=errors or [],
        repairs_attempted=repairs_attempted,
        repairs_successful=repairs_attempted if all_passed else 0,
        to_dict=lambda: {
            "all_passed": all_passed,
            "syntax_ok": syntax_ok,
            "imports_ok": imports_ok,
            "errors": errors or [],
        },
    )


@pytest.fixture
def paper_context():
    return PaperContext(
        title="E2E Paper",
        abstract="Abstract",
        method_section="Method",
    )


@pytest.mark.asyncio
async def test_happy_path_completes_without_retries(tmp_path, paper_context):
    agent = _legacy_agent_with_common_nodes()
    agent.verification_node.run = AsyncMock(
        return_value=NodeResult.ok(_report(all_passed=True, syntax_ok=True, imports_ok=True))
    )

    result = await agent.reproduce_from_paper(paper_context, output_dir=tmp_path)

    assert result.status is ReproPhase.COMPLETED
    assert result.retry_count == 0
    assert (tmp_path / "main.py").exists()


@pytest.mark.asyncio
async def test_retry_path_records_repairs_and_still_completes(tmp_path, paper_context):
    agent = _legacy_agent_with_common_nodes()
    agent.verification_node.run = AsyncMock(
        return_value=NodeResult.ok(
            _report(
                all_passed=False,
                syntax_ok=True,
                imports_ok=True,
                errors=["SyntaxError: invalid syntax"],
                repairs_attempted=1,
            )
        )
    )

    result = await agent.reproduce_from_paper(paper_context, output_dir=tmp_path)

    assert result.status is ReproPhase.COMPLETED
    assert result.retry_count == 1
    assert result.verification["syntax_ok"] is True


@pytest.mark.asyncio
async def test_failure_path_surfaces_verification_errors(tmp_path, paper_context):
    agent = _legacy_agent_with_common_nodes()
    agent.verification_node.run = AsyncMock(
        return_value=NodeResult.ok(
            _report(
                all_passed=False,
                syntax_ok=False,
                imports_ok=False,
                errors=["Fatal Error"],
                repairs_attempted=1,
            )
        )
    )

    result = await agent.reproduce_from_paper(paper_context, output_dir=tmp_path)

    assert result.status is ReproPhase.FAILED
    assert result.retry_count == 1
    assert "Fatal Error" in result.errors
