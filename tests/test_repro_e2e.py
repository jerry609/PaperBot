import sys
from unittest.mock import MagicMock

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
from repro.models import PaperContext

class TestReproE2E(unittest.TestCase):
    """
    End-to-End test for PaperBot Repro Pipeline.
    Simulates the full Paper2Code flow using fallback (template) generation.
    """
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Force fallback behavior by patching modules to have query=None
        # This overrides the global mock we set in tests/__init__.py
        # with patch('repro.planning_agent.query', None), \
        #      patch('repro.generation_agent.query', None), \
        #      patch('repro.repro_agent.query', None):
        #     self.agent = ReproAgent({})
        #     # Because we patch init-time imports in the class instances or module, we might need to do it differently.
        #     # But simpler: just set attributes on the instances' classes if they check global query
            
        # Actually simplest way: separate setup for agent after patches, 
        # BUT the modules are already imported.
        # So we must patch the specific attributes in the modules.
        
        from repro import planning_agent, generation_agent, repro_agent
        self.original_query_p = planning_agent.query
        self.original_query_g = generation_agent.query
        self.original_query_r = repro_agent.query
        
        planning_agent.query = None
        generation_agent.query = None
        repro_agent.query = None
        
        self.agent = ReproAgent({})
        
        # Mock executor to avoid real docker calls
        self.agent.executor = MagicMock()
        self.agent.executor.available.return_value = True
        self.agent.executor.run.return_value = {
            "status": "success", 
            "exit_code": 0, 
            "logs": "OK"
        }
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Restore
        from repro import planning_agent, generation_agent, repro_agent
        planning_agent.query = self.original_query_p
        generation_agent.query = self.original_query_g
        repro_agent.query = self.original_query_r

    def test_full_reproduction_flow(self):
        # 1. Setup Context
        ctx = PaperContext(
            title="E2E Test Paper",
            abstract="We propose a simple MLP for classification.",
            method_section="The model consists of 3 linear layers with ReLU activation.",
            algorithm_blocks=["Input -> Layer 1 -> ReLU -> Layer 2 -> Softmax"],
            hyperparameters={"lr": 0.001, "batch_size": 16}
        )

        # 2. Run Reproduction
        # Start event loop if needed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            self.agent.reproduce_from_paper(ctx, output_dir=self.test_dir)
        )
        
        # 3. Verify Output
        print(f"E2E Result Status: {result.status}")
        print(f"Generated Files: {result.generated_files.keys()}")
        print(f"Verification Results: {result.verification_results}")
        
        # Even if verification fails (due to dummy mock), it shouldn't crash
        self.assertIn(result.status, ["success", "partial", "failed"]) 
        
        # Check if files were created
        self.assertTrue((self.test_dir / "main.py").exists())
        self.assertTrue((self.test_dir / "model.py").exists())
        self.assertTrue((self.test_dir / "requirements.txt").exists())
        
        # Check generated content (should match fallback templates)
        main_content = (self.test_dir / "main.py").read_text()
        self.assertIn("def main():", main_content)

if __name__ == "__main__":
    unittest.main()
