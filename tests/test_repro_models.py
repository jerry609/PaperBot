import sys
from unittest.mock import MagicMock

# Mock dependencies before importing modules under test
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["claude_agent_sdk"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
from repro.models import PaperContext, ReproductionPlan, ImplementationSpec, VerificationResult, VerificationStep

class TestReproModels(unittest.TestCase):
    def test_paper_context_to_prompt(self):
        ctx = PaperContext(
            title="Test Paper",
            abstract="This is an abstract.",
            method_section="Method details.",
            algorithm_blocks=["Block 1"],
            hyperparameters={"lr": 0.01}
        )
        prompt = ctx.to_prompt_context()
        self.assertIn("## Paper: Test Paper", prompt)
        self.assertIn("This is an abstract.", prompt)
        self.assertIn("Method details.", prompt)
        self.assertIn("Block 1", prompt)
        self.assertIn("'lr': 0.01", prompt)

    def test_reproduction_plan_to_prompt(self):
        plan = ReproductionPlan(
            project_name="test_proj",
            description="A test project",
            file_structure={"main.py": "entry"},
            entry_point="main.py",
            dependencies=["numpy"],
            key_components=["Model"]
        )
        prompt = plan.to_prompt_context()
        self.assertIn("## Reproduction Plan: test_proj", prompt)
        self.assertIn("A test project", prompt)
        self.assertIn("`main.py`: entry", prompt)
        self.assertIn("Entry Point", prompt)
        self.assertIn("numpy", prompt)

    def test_implementation_spec_to_prompt(self):
        spec = ImplementationSpec(
            model_type="transformer",
            layers=[{"type": "Linear"}],
            optimizer="adam"
        )
        prompt = spec.to_prompt_context()
        self.assertIn("Model Type:** transformer", prompt)
        self.assertIn("Optimizer:** adam", prompt)
        self.assertIn("Linear", prompt)

    def test_verification_result_to_dict(self):
        res = VerificationResult(
            step=VerificationStep.SYNTAX_CHECK,
            passed=True,
            message="All good",
            duration_sec=1.5
        )
        d = res.to_dict()
        self.assertEqual(d["step"], "syntax_check")
        self.assertTrue(d["passed"])
        self.assertEqual(d["message"], "All good")
        self.assertEqual(d["duration_sec"], 1.5)

if __name__ == "__main__":
    unittest.main()
