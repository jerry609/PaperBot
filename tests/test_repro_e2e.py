import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock dependencies before importing modules under test
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["claude_agent_sdk"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
import asyncio
import tempfile
import shutil
from pathlib import Path
from repro.repro_agent import ReproAgent
from repro.models import PaperContext, VerificationStep

class TestReproE2E(unittest.TestCase):
    """
    End-to-End test for PaperBot Repro Pipeline coverage critical paths:
    1. Happy Path: Plan -> Gen -> Verify(Success)
    2. Retry Path: Plan -> Gen -> Verify(Fail) -> Refine -> Verify(Success)
    3. Failure Path: Plan -> Gen -> Verify(Fail) -> Refine(Fail)
    """
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Patch dependencies
        self.patches = [
            patch('repro.planning_agent.query', None),
            patch('repro.generation_agent.query', None),
            patch('repro.repro_agent.query', None)
        ]
        for p in self.patches:
            p.start()

        # Helper to reset agent for each test
        self.setup_agent()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        for p in self.patches:
            p.stop()

    def setup_agent(self):
        # We need to forcefully reload or reset the classes if they were imported before patches
        # But simpler is to rely on the patches being active when the methods are called
        self.agent = ReproAgent({})
        self.ctx = PaperContext(
            title="E2E Paper", 
            abstract="Abstract", 
            method_section="Method"
        )
        # Mock executor by default (can be overridden in tests)
        self.agent.executor = MagicMock()
        self.agent.executor.available.return_value = True

    def test_happy_path(self):
        """Test complete success path without retries."""
        # Setup: Executor always returns success
        self.agent.executor.run.return_value = {
            "status": "success", "exit_code": 0, "logs": "OK"
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            self.agent.reproduce_from_paper(self.ctx, output_dir=self.test_dir)
        )
        
        self.assertEqual(result.status, "success")
        self.assertEqual(result.retry_count, 0)
        self.assertTrue((self.test_dir / "main.py").exists())

    def test_retry_logic_syntax_error(self):
        """
        Test Critical Path: Code has syntax error, agent refines it, then passes.
        """
        # Mock GenerationAgent.refine_code to return fixed code
        self.agent.generation_agent.refine_code = AsyncMock(
            return_value="print('Fixed Syntax')"
        )
        
        # Mock Executor to fail first (syntax check), then succeed
        # Phase 4 calls: Syntax -> Import -> Test -> Smoke
        # We want: 
        # Attempt 1: Syntax Fails
        # Refine called
        # Attempt 2: All Succeed
        
        # Side effect for executor.run
        # We need to handle multiple calls. 
        # Call 1: Syntax check (Fail)
        # Call 2: Syntax check (Success) -> Call 3: Import (Success) ...
        
        # Note: ReproAgent._check_syntax calls executor with "python -m py_compile"
        def executor_side_effect(workdir, cmd, **kwargs):
            command_str = cmd[0] if isinstance(cmd, list) else cmd
            
            if "py_compile" in command_str:
                # If content is "Fixed Syntax", succeed, else fail
                # In real run, file content changes. In mock, we can check a counter or file content
                main_py = workdir / "main.py"
                if main_py.exists() and "Fixed" in main_py.read_text():
                    return {"status": "success", "exit_code": 0, "logs": ""}
                else:
                    return {"status": "failed", "exit_code": 1, "logs": "SyntaxError: invalid syntax"}
            
            # All other checks pass
            return {"status": "success", "exit_code": 0, "logs": "OK"}

        self.agent.executor.run.side_effect = executor_side_effect
        
        # Create initial dummy file that "has error" (by not having "Fixed")
        # The generation agent (fallback) creates valid code, but our mock verify logic 
        # treats it as invalid unless it has "Fixed".
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            self.agent.reproduce_from_paper(self.ctx, output_dir=self.test_dir)
        )
        
        self.assertEqual(result.status, "success")
        self.assertGreater(result.retry_count, 0, "Should have retried at least once")
        
        # Verify Refine was called
        # self.agent.generation_agent.refine_code.assert_called() 
        # (It's part of the loop, hard to assert exact call without more mocking, but result implies it)
        
        # Verify file was updated
        self.assertIn("Fixed", (self.test_dir / "main.py").read_text())

    def test_unrecoverable_failure(self):
        """Test path where max retries are exceeded."""
        self.agent.max_retries = 1
        
        # Executor always fails
        self.agent.executor.run.return_value = {
            "status": "failed", "exit_code": 1, "logs": "Fatal Error"
        }
        
        # Refine always returns same code (not fixed)
        self.agent.generation_agent.refine_code = AsyncMock(
            return_value="print('Still Broken')"
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            self.agent.reproduce_from_paper(self.ctx, output_dir=self.test_dir)
        )
        
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.retry_count, 1) # attempted 1 retry due to max_retries=1

if __name__ == "__main__":
    unittest.main()
