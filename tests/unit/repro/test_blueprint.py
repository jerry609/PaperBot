# tests/unit/repro/test_blueprint.py
"""
Unit tests for Blueprint and related models.
"""

import sys
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
from repro.models import (
    Blueprint,
    AlgorithmSpec,
    PaperContext,
    ReproductionPlan,
    ImplementationSpec,
    EnvironmentSpec,
    ReproductionResult,
    ReproPhase,
)


class TestAlgorithmSpec(unittest.TestCase):
    """Tests for AlgorithmSpec dataclass."""

    def test_creation(self):
        spec = AlgorithmSpec(
            name="Attention",
            pseudocode="1. Compute Q, K, V\n2. Apply softmax",
            inputs=["query", "key", "value"],
            outputs=["attention_output"],
        )

        self.assertEqual(spec.name, "Attention")
        self.assertEqual(len(spec.inputs), 3)
        self.assertEqual(len(spec.outputs), 1)


class TestBlueprint(unittest.TestCase):
    """Tests for Blueprint dataclass."""

    def test_creation_minimal(self):
        blueprint = Blueprint(
            architecture_type="transformer",
            domain="nlp",
        )

        self.assertEqual(blueprint.architecture_type, "transformer")
        self.assertEqual(blueprint.domain, "nlp")

    def test_creation_full(self):
        algo = AlgorithmSpec(
            name="Attention",
            pseudocode="Attention mechanism",
            inputs=["Q", "K", "V"],
            outputs=["out"],
        )

        blueprint = Blueprint(
            architecture_type="transformer",
            domain="nlp",
            core_algorithms=[algo],
            key_hyperparameters={"d_model": 512, "n_heads": 8},
            loss_functions=["cross_entropy"],
            optimization_strategy="Adam with warmup",
            framework_hints=["pytorch"],
        )

        self.assertEqual(len(blueprint.core_algorithms), 1)
        self.assertEqual(blueprint.key_hyperparameters["d_model"], 512)
        self.assertIn("pytorch", blueprint.framework_hints)

    def test_to_compressed_context(self):
        blueprint = Blueprint(
            architecture_type="transformer",
            domain="nlp",
            key_hyperparameters={"d_model": 512},
            loss_functions=["cross_entropy"],
        )

        context = blueprint.to_compressed_context()

        self.assertIn("transformer", context)
        self.assertIn("nlp", context)
        self.assertIn("d_model", context)


class TestPaperContext(unittest.TestCase):
    """Tests for PaperContext dataclass."""

    def test_minimal_creation(self):
        ctx = PaperContext(
            title="Test Paper",
            abstract="This is a test.",
        )

        self.assertEqual(ctx.title, "Test Paper")
        self.assertEqual(ctx.abstract, "This is a test.")

    def test_full_creation(self):
        ctx = PaperContext(
            title="Attention Is All You Need",
            abstract="We propose a new architecture...",
            method_section="The Transformer model...",
            algorithm_blocks=["Algorithm 1: Attention"],
            hyperparameters={"d_model": 512, "n_heads": 8},
        )

        self.assertEqual(len(ctx.algorithm_blocks), 1)
        self.assertEqual(ctx.hyperparameters["d_model"], 512)

    def test_to_prompt_context(self):
        ctx = PaperContext(
            title="Test Paper",
            abstract="Abstract text here.",
            method_section="Method details.",
            hyperparameters={"lr": 0.001},
        )

        prompt = ctx.to_prompt_context()

        self.assertIn("## Paper: Test Paper", prompt)
        self.assertIn("Abstract text here", prompt)
        self.assertIn("Method details", prompt)
        self.assertIn("lr", prompt)


class TestReproductionPlan(unittest.TestCase):
    """Tests for ReproductionPlan dataclass."""

    def test_creation(self):
        plan = ReproductionPlan(
            project_name="my_project",
            description="A test project",
            file_structure={"main.py": "entry", "model.py": "model"},
            entry_point="main.py",
            dependencies=["torch", "numpy"],
            key_components=["Model", "Trainer"],
        )

        self.assertEqual(plan.project_name, "my_project")
        self.assertEqual(len(plan.file_structure), 2)
        self.assertIn("torch", plan.dependencies)

    def test_to_prompt_context(self):
        plan = ReproductionPlan(
            project_name="test_proj",
            description="Test description",
            file_structure={"main.py": "entry point"},
            entry_point="main.py",
            dependencies=["numpy"],
            key_components=["Model"],
        )

        prompt = plan.to_prompt_context()

        self.assertIn("test_proj", prompt)
        self.assertIn("main.py", prompt)
        self.assertIn("numpy", prompt)


class TestImplementationSpec(unittest.TestCase):
    """Tests for ImplementationSpec dataclass."""

    def test_creation(self):
        spec = ImplementationSpec(
            model_type="transformer",
            layers=[{"type": "Linear", "in": 512, "out": 512}],
            optimizer="adam",
            learning_rate=0.001,
            batch_size=32,
        )

        self.assertEqual(spec.model_type, "transformer")
        self.assertEqual(len(spec.layers), 1)
        self.assertEqual(spec.learning_rate, 0.001)

    def test_to_prompt_context(self):
        spec = ImplementationSpec(
            model_type="cnn",
            optimizer="sgd",
            batch_size=64,
        )

        prompt = spec.to_prompt_context()

        self.assertIn("cnn", prompt)
        self.assertIn("sgd", prompt)


class TestEnvironmentSpec(unittest.TestCase):
    """Tests for EnvironmentSpec dataclass."""

    def test_defaults(self):
        spec = EnvironmentSpec()

        self.assertEqual(spec.python_version, "3.10")
        self.assertEqual(spec.base_image, "python:3.10-slim")

    def test_generate_dockerfile(self):
        spec = EnvironmentSpec(
            python_version="3.9",
            pip_requirements=["torch", "numpy"],
        )

        dockerfile = spec.generate_dockerfile()

        self.assertIn("FROM python", dockerfile)
        self.assertIn("pip install", dockerfile)


class TestReproductionResult(unittest.TestCase):
    """Tests for ReproductionResult dataclass."""

    def test_default_values(self):
        result = ReproductionResult(paper_title="Test Paper")

        self.assertEqual(result.paper_title, "Test Paper")
        self.assertEqual(result.status, ReproPhase.PLANNING)
        self.assertEqual(result.phases_completed, [])
        self.assertEqual(result.generated_files, {})

    def test_compute_score_with_results(self):
        from repro.models import VerificationResult, VerificationStep

        result = ReproductionResult(
            paper_title="Test",
            status=ReproPhase.COMPLETED,
        )
        result.verification_results = [
            VerificationResult(step=VerificationStep.SYNTAX_CHECK, passed=True, message="OK"),
            VerificationResult(step=VerificationStep.IMPORT_CHECK, passed=True, message="OK"),
        ]

        score = result.compute_score()
        self.assertGreater(score, 0)

    def test_compute_score_empty(self):
        result = ReproductionResult(
            paper_title="Test",
            status=ReproPhase.FAILED,
        )

        score = result.compute_score()
        self.assertEqual(score, 0)

    def test_to_dict(self):
        result = ReproductionResult(
            paper_title="Test Paper",
            status=ReproPhase.COMPLETED,
            phases_completed=["planning"],
            generated_files={"main.py": "# code"},
            retry_count=1,
        )

        d = result.to_dict()

        self.assertEqual(d["paper_title"], "Test Paper")
        self.assertEqual(d["status"], "completed")
        self.assertIn("planning", d["phases_completed"])
        self.assertEqual(d["retry_count"], 1)


if __name__ == "__main__":
    unittest.main()
