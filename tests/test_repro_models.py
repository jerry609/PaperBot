"""Compatibility tests for repro model helpers."""

from __future__ import annotations

from paperbot.repro import (
    ImplementationSpec,
    PaperContext,
    ReproductionPlan,
    ReproductionResult,
    ReproPhase,
    VerificationResult,
    VerificationStep,
)


def test_paper_context_to_prompt_contains_sections():
    context = PaperContext(
        title="Test Paper",
        abstract="This is an abstract.",
        method_section="Method details.",
        algorithm_blocks=["Block 1"],
        hyperparameters={"lr": 0.01},
    )

    prompt = context.to_prompt_context()

    assert "## Paper: Test Paper" in prompt
    assert "This is an abstract." in prompt
    assert "Method details." in prompt
    assert "Block 1" in prompt
    assert "'lr': 0.01" in prompt


def test_reproduction_plan_to_prompt_context_lists_files():
    plan = ReproductionPlan(
        project_name="test_proj",
        description="A test project",
        file_structure={"main.py": "entry"},
        entry_point="main.py",
        dependencies=["numpy"],
        key_components=["Model"],
    )

    prompt = plan.to_prompt_context()

    assert "## Reproduction Plan: test_proj" in prompt
    assert "`main.py`: entry" in prompt
    assert "numpy" in prompt


def test_implementation_spec_to_prompt_context_includes_optimizer():
    spec = ImplementationSpec(
        model_type="transformer",
        layers=[{"type": "Linear"}],
        optimizer="adam",
    )

    prompt = spec.to_prompt_context()

    assert "Model Type:** transformer" in prompt
    assert "Optimizer:** adam" in prompt
    assert "Linear" in prompt


def test_verification_result_and_reproduction_result_to_dict():
    verification = VerificationResult(
        step=VerificationStep.SYNTAX_CHECK,
        passed=True,
        message="All good",
        duration_sec=1.5,
    )
    result = ReproductionResult(
        paper_title="Test Paper",
        status=ReproPhase.COMPLETED,
        phases_completed=["planning"],
        generated_files={"main.py": "# code"},
        retry_count=1,
    )
    result.verification_results = [verification]
    result.compute_score()

    verification_payload = verification.to_dict()
    result_payload = result.to_dict()

    assert verification_payload["step"] == "syntax_check"
    assert verification_payload["passed"] is True
    assert result_payload["status"] == "completed"
    assert result_payload["retry_count"] == 1
    assert result_payload["generated_files"] == ["main.py"]
