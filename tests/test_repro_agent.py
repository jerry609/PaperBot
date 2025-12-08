import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock dependencies before importing modules under test
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["claude_agent_sdk"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
import asyncio
from pathlib import Path
from repro.repro_agent import ReproAgent
from repro.models import PaperContext, ReproductionResult, ReproPhase

class DummyExecutor:
    def __init__(self, status="success"):
        self._status = status

    def available(self):
        return True

    def run(self, workdir: Path, commands, timeout_sec: int = 300, cache_dir=None):
        return {
            "status": self._status,
            "exit_code": 0 if self._status == "success" else 1,
            "logs": "ok",
            "duration_sec": 1.0,
        }

class TestReproAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ReproAgent({})
        self.agent.executor = DummyExecutor()
        self.ctx = PaperContext(title="Test", abstract="Abs")

    def test_legacy_run(self):
        # Test backward compatibility
        with patch.object(self.agent, 'generate_plan', new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = ["echo hello"]
            res = asyncio.run(self.agent.run(Path(".")))
            self.assertEqual(res["status"], "success")

    def test_paper2code_flow(self):
        # Test multi-phase reproduction
        # Mock sub-agents to avoid LLM calls
        self.agent.planning_agent.generate_plan = AsyncMock(return_value=MagicMock(
            file_structure={"main.py": "purpose"}
        ))
        self.agent.planning_agent.generate_spec = AsyncMock(return_value=MagicMock())
        self.agent.generation_agent.generate_code = AsyncMock(return_value={"main.py": "print('hi')"})
        
        # Also mock internal verification to avoid docker calls that might fail if not fully mocked
        self.agent._verify_with_retry = AsyncMock(return_value=([], 0))
        
        # Run
        res = asyncio.run(self.agent.reproduce_from_paper(self.ctx))
        
        self.assertEqual(res.status, "failed") # Defaults to failed if score is low (verification skipped means 0 score)
        self.assertIn(ReproPhase.PLANNING.value, res.phases_completed)
        self.assertIn(ReproPhase.VERIFICATION.value, res.phases_completed)
        self.assertTrue(res.generated_files)

if __name__ == "__main__":
    unittest.main()
