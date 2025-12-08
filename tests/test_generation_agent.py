import sys
from unittest.mock import MagicMock

# Mock dependencies before importing modules under test
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["claude_agent_sdk"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
from unittest.mock import patch
from repro.generation_agent import GenerationAgent
from repro.models import PaperContext, ReproductionPlan, ImplementationSpec

class TestGenerationAgent(unittest.TestCase):
    def setUp(self):
        self.agent = GenerationAgent()
        self.ctx = PaperContext(title="Test", abstract="Abs")
        self.plan = ReproductionPlan(
            project_name="test",
            description="desc",
            file_structure={"main.py": "Main file"},
            entry_point="main.py"
        )
        self.spec = ImplementationSpec(model_type="mlp")

    def test_fallback_template(self):
        # Test fallback templates
        code = self.agent._fallback_template("main.py", "Main", self.plan, self.spec)
        self.assertIn("import argparse", code)
        
        code = self.agent._fallback_template("unknown.py", "Unknown", self.plan, self.spec)
        self.assertIn("# TODO: Implement", code)

    def test_generate_requirements(self):
        reqs = self.agent._generate_requirements(self.plan)
        self.assertIn("torch", reqs)
        self.assertIn("numpy", reqs)

    def test_clean_code_response(self):
        raw = "```python\nprint('hello')\n```"
        clean = self.agent._clean_code_response(raw)
        self.assertEqual(clean, "print('hello')")
        
        raw2 = "print('hello')"
        clean2 = self.agent._clean_code_response(raw2)
        self.assertEqual(clean2, "print('hello')")

if __name__ == "__main__":
    unittest.main()
