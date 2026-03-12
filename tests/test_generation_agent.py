"""Compatibility tests for the current repro code-generation layer."""

from __future__ import annotations

from repro import Blueprint, GenerationNode, ImplementationSpec, ReproductionPlan


def test_generation_node_fallback_templates_cover_known_and_generic_files():
    node = GenerationNode()
    plan = ReproductionPlan(
        project_name="test",
        description="desc",
        file_structure={"main.py": "Main file"},
        entry_point="main.py",
        dependencies=["torch", "numpy"],
    )
    spec = ImplementationSpec(model_type="mlp")

    main_code = node._fallback_template("main.py", "Main file", plan, spec)
    generic_code = node._fallback_template("unknown.py", "Unknown file", plan, spec)

    assert "import argparse" in main_code
    assert "# TODO: Implement unknown" in generic_code


def test_generation_node_generate_requirements_uses_plan_dependencies():
    node = GenerationNode()
    plan = ReproductionPlan(
        project_name="test",
        description="desc",
        dependencies=["torch", "numpy"],
    )

    requirements = node._generate_requirements(plan)

    assert requirements.splitlines() == ["torch", "numpy"]


def test_generation_node_clean_code_strips_markdown_fences():
    node = GenerationNode()

    assert node._clean_code("```python\nprint('hello')\n```") == "print('hello')"
    assert node._clean_code("print('hello')") == "print('hello')"


def test_generation_node_retrieves_patterns_from_blueprint_hints():
    node = GenerationNode(use_rag=True)
    blueprint = Blueprint(
        architecture_type="transformer",
        framework_hints=["pytorch"],
        domain="nlp",
    )

    patterns = node._retrieve_patterns("model.py", "Model implementation", blueprint)

    assert patterns
    assert any(pattern.name for pattern in patterns)
