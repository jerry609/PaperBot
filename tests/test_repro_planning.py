import sys
from unittest.mock import MagicMock, patch

# Mock dependencies before importing modules under test
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["claude_agent_sdk"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
from repro.planning_agent import PlanningAgent
from repro.models import PaperContext, ReproductionPlan, ImplementationSpec

class TestPlanningAgent(unittest.TestCase):
    def setUp(self):
        with patch('repro.planning_agent.query', None):
            self.agent = PlanningAgent()
        self.ctx = PaperContext(
            title="Test Paper",
            abstract="Abstract",
            method_section="Method"
        )

    def test_fallback_plan(self):
        # Test fallback logic when LLM is mocked/unavailable
        with patch('repro.planning_agent.query', None):
            plan = self.agent._fallback_plan(self.ctx)
            self.assertIsInstance(plan, ReproductionPlan)
            self.assertEqual(plan.entry_point, "main.py")
            self.assertIn("main.py", plan.file_structure)

    def test_fallback_spec(self):
        with patch('repro.planning_agent.query', None):
            spec = self.agent._fallback_spec()
            self.assertIsInstance(spec, ImplementationSpec)
            self.assertEqual(spec.optimizer, "adam")
    
    def test_parse_plan_json(self):
        json_str = """
        {
            "project_name": "test",
            "description": "desc",
            "file_structure": {"main.py": "entry"},
            "entry_point": "main.py",
            "dependencies": ["torch"],
            "key_components": ["Model"],
            "estimated_complexity": "low"
        }
        """
        plan = self.agent._parse_plan(json_str, self.ctx)
        self.assertEqual(plan.project_name, "test")
        self.assertEqual(plan.dependencies, ["torch"])

    def test_parse_plan_invalid(self):
        # Should return fallback on invalid JSON
        plan = self.agent._parse_plan("invalid json", self.ctx)
        self.assertEqual(plan.description, f"Reproduction of: {self.ctx.title}")

if __name__ == "__main__":
    unittest.main()
