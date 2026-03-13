"""Pytest coverage for repro data models and blueprint helpers."""

from __future__ import annotations

from repro import (
    AlgorithmSpec,
    Blueprint,
    EnvironmentSpec,
    ImplementationSpec,
    PaperContext,
    ReproductionPlan,
    ReproductionResult,
    ReproPhase,
    VerificationResult,
    VerificationStep,
)


def test_algorithm_spec_defaults():
    spec = AlgorithmSpec(name="Attention", inputs=["query", "key"], outputs=["value"])

    assert spec.name == "Attention"
    assert spec.complexity == ""
    assert spec.inputs == ["query", "key"]
    assert spec.outputs == ["value"]


def test_blueprint_to_compressed_context_includes_high_signal_sections():
    blueprint = Blueprint(
        architecture_type="transformer",
        module_hierarchy={"model": ["encoder", "decoder"]},
        key_hyperparameters={"d_model": 512, "n_heads": 8},
        loss_functions=["cross_entropy"],
        framework_hints=["pytorch"],
        paper_title="Attention Is All You Need",
        paper_year=2017,
        domain="nlp",
    )

    context = blueprint.to_compressed_context()

    assert "Attention Is All You Need" in context
    assert "transformer" in context
    assert "encoder" in context
    assert "d_model" in context
    assert "cross_entropy" in context


def test_blueprint_round_trips_through_dict_serialization():
    original = Blueprint(
        architecture_type="gnn",
        module_hierarchy={"model": ["message_passing"]},
        data_flow=[("input", "encoder")],
        core_algorithms=[AlgorithmSpec(name="MessagePassing", pseudocode="aggregate")],
        key_hyperparameters={"hidden_dim": 128},
        input_output_spec={"input": "graph", "output": "logits"},
        framework_hints=["pytorch"],
        paper_title="Graph Networks",
        paper_year=2020,
        domain="cv",
    )

    restored = Blueprint.from_dict(original.to_dict())

    assert restored.architecture_type == "gnn"
    assert restored.module_hierarchy == {"model": ["message_passing"]}
    assert restored.core_algorithms[0].name == "MessagePassing"
    assert restored.input_output_spec["output"] == "logits"


def test_paper_context_to_prompt_context_renders_sections():
    context = PaperContext(
        title="Test Paper",
        abstract="This is an abstract.",
        method_section="Method details.",
        algorithm_blocks=["for step in range(n): pass"],
        hyperparameters={"lr": 0.01},
    )

    prompt = context.to_prompt_context()

    assert "## Paper: Test Paper" in prompt
    assert "This is an abstract." in prompt
    assert "Method details." in prompt
    assert "for step in range(n): pass" in prompt
    assert "'lr': 0.01" in prompt


def test_reproduction_plan_to_prompt_context_lists_files_and_dependencies():
    plan = ReproductionPlan(
        project_name="test_proj",
        description="A test project",
        file_structure={"main.py": "entry point", "model.py": "model code"},
        entry_point="main.py",
        dependencies=["numpy", "torch"],
        key_components=["Model", "Trainer"],
    )

    prompt = plan.to_prompt_context()

    assert "test_proj" in prompt
    assert "`main.py`: entry point" in prompt
    assert "torch" in prompt
    assert "Trainer" in prompt


def test_implementation_spec_to_prompt_context_includes_layers_and_optimizer():
    spec = ImplementationSpec(
        model_type="transformer",
        layers=[{"type": "Linear", "in": 512, "out": 512}],
        optimizer="adamw",
        learning_rate=3e-4,
        batch_size=16,
    )

    prompt = spec.to_prompt_context()

    assert "transformer" in prompt
    assert "adamw" in prompt
    assert "Linear" in prompt
    assert "Batch Size" in prompt


def test_environment_spec_generates_dockerfile_and_conda_yaml():
    spec = EnvironmentSpec(
        python_version="3.9",
        pytorch_version="2.2.0",
        pip_requirements=["numpy", "pytest"],
    )

    dockerfile = spec.generate_dockerfile()
    conda_yaml = spec.generate_conda_yaml()

    assert dockerfile.startswith("FROM ")
    assert "pip install -r requirements.txt" in dockerfile
    assert "python=3.9" in conda_yaml
    assert "numpy" in conda_yaml


def test_reproduction_result_compute_score_and_to_dict():
    result = ReproductionResult(
        paper_title="Test Paper",
        status=ReproPhase.COMPLETED,
        generated_files={"main.py": "print('hi')"},
        phases_completed=["planning", "generation"],
        retry_count=1,
    )
    result.verification_results = [
        VerificationResult(step=VerificationStep.SYNTAX_CHECK, passed=True, message="OK"),
        VerificationResult(step=VerificationStep.IMPORT_CHECK, passed=True, message="OK"),
    ]

    score = result.compute_score()
    payload = result.to_dict()

    assert score == 50
    assert payload["paper_title"] == "Test Paper"
    assert payload["status"] == "completed"
    assert payload["generated_files"] == ["main.py"]
    assert payload["retry_count"] == 1


def test_reproduction_result_score_is_zero_without_verification():
    result = ReproductionResult(paper_title="Test Paper", status=ReproPhase.FAILED)

    assert result.compute_score() == 0
