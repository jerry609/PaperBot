# tests/integration/test_repro_deepcode.py
"""
Integration tests for DeepCode-enhanced repro pipeline.
Tests the full flow from paper context to code generation.
"""

import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock external dependencies
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
import asyncio
import tempfile
from pathlib import Path

from repro import (
    ReproAgent,
    PaperContext,
    Blueprint,
    CodeMemory,
    CodeKnowledgeBase,
    Orchestrator,
    OrchestratorConfig,
    ReproPhase,
)
from repro.nodes import BlueprintDistillationNode, PlanningNode, GenerationNode
from repro.agents import AgentResult, AgentStatus


class TestDeepCodePipelineIntegration(unittest.TestCase):
    """Integration tests for the full DeepCode pipeline."""

    def setUp(self):
        self.paper_context = PaperContext(
            title="Attention Is All You Need",
            abstract="""
            We propose a new simple network architecture, the Transformer,
            based solely on attention mechanisms, dispensing with recurrence
            and convolutions entirely. The dominant sequence transduction
            models are based on complex recurrent or convolutional neural
            networks that include an encoder and a decoder.
            """,
            method_section="""
            The Transformer follows an encoder-decoder structure using
            stacked self-attention and point-wise, fully connected layers.
            The encoder maps an input sequence to a sequence of continuous
            representations. The decoder generates an output sequence.
            """,
            hyperparameters={"d_model": 512, "n_heads": 8, "n_layers": 6},
        )

    def test_memory_and_rag_integration(self):
        """Test that CodeMemory and RAG work together."""
        memory = CodeMemory(max_context_tokens=2000)
        kb = CodeKnowledgeBase.from_builtin()

        # Add a config file
        config_code = '''
class Config:
    d_model = 512
    n_heads = 8
'''
        memory.add_file("config.py", config_code, purpose="Configuration")

        # Search for related patterns
        patterns = kb.search("transformer attention", k=2)
        self.assertGreater(len(patterns), 0)

        # Get context for model generation
        context = memory.get_relevant_context("model.py", "Transformer model")
        self.assertIn("config.py", context)

    def test_blueprint_to_plan_flow(self):
        """Test Blueprint distillation to planning flow."""
        blueprint = Blueprint(
            architecture_type="transformer",
            domain="nlp",
            key_hyperparameters={"d_model": 512, "n_heads": 8},
            loss_functions=["cross_entropy"],
            framework_hints=["pytorch"],
        )

        # Blueprint should be convertible to planning context
        context = blueprint.to_compressed_context()

        self.assertIn("transformer", context)
        self.assertIn("nlp", context)
        self.assertIn("d_model", context)

    def test_generation_with_memory(self):
        """Test that GenerationNode uses CodeMemory correctly."""
        node = GenerationNode(max_context_tokens=2000, use_rag=True)

        # Verify memory is initialized
        self.assertIsNotNone(node.memory)
        self.assertIsNotNone(node.knowledge_base)

        # Verify token budget is respected
        self.assertEqual(node.memory.max_context_tokens, 2000)

    def test_orchestrator_initialization(self):
        """Test Orchestrator initializes all agents correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(
                max_repair_loops=2,
                output_dir=Path(tmpdir),
                use_rag=True,
            )

            orchestrator = Orchestrator(config=config)

            self.assertIsNotNone(orchestrator.planning_agent)
            self.assertIsNotNone(orchestrator.coding_agent)
            self.assertIsNotNone(orchestrator.verification_agent)
            self.assertIsNotNone(orchestrator.debugging_agent)

    def test_repro_agent_mode_switching(self):
        """Test ReproAgent can switch between legacy and orchestrator modes."""
        # Legacy mode
        agent_legacy = ReproAgent({"use_orchestrator": False})
        self.assertFalse(agent_legacy.use_orchestrator)

        # Orchestrator mode
        agent_orch = ReproAgent({"use_orchestrator": True})
        self.assertTrue(agent_orch.use_orchestrator)

    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked LLM calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(
                max_repair_loops=1,
                output_dir=Path(tmpdir),
            )

            orchestrator = Orchestrator(config=config)

            # Mock all agent runs
            mock_plan = MagicMock()
            mock_plan.file_structure = {"main.py": "entry", "model.py": "model"}

            async def mock_planning(context):
                context["plan"] = mock_plan
                context["blueprint"] = Blueprint(
                    architecture_type="transformer",
                    domain="nlp",
                )
                return AgentResult.success(data={"plan": mock_plan})

            async def mock_coding(context):
                context["generated_files"] = {
                    "main.py": "print('hello')",
                    "model.py": "class Model: pass",
                }
                # Write files to disk
                output_dir = context.get("output_dir", Path(tmpdir))
                for name, content in context["generated_files"].items():
                    (output_dir / name).write_text(content)
                return AgentResult.success(data={"generated_files": context["generated_files"]})

            mock_report = MagicMock()
            mock_report.all_passed = True
            mock_report.to_dict.return_value = {"syntax_ok": True, "imports_ok": True}

            async def mock_verification(context):
                context["verification_report"] = mock_report
                return AgentResult.success(data={"report": mock_report})

            orchestrator.planning_agent.run = mock_planning
            orchestrator.coding_agent.run = mock_coding
            orchestrator.verification_agent.run = mock_verification

            result = asyncio.run(orchestrator.run(self.paper_context))

            self.assertEqual(result.status, ReproPhase.COMPLETED)
            self.assertIn("planning", result.phases_completed)
            self.assertIn("generation", result.phases_completed)

    def test_dependency_ordering(self):
        """Test that files are generated in correct dependency order."""
        memory = CodeMemory()

        file_structure = {
            "main.py": "entry point",
            "trainer.py": "training logic",
            "model.py": "neural network",
            "config.py": "configuration",
            "data.py": "data loading",
            "utils.py": "utilities",
        }

        order = memory.compute_generation_order(file_structure)

        # Config should come before model
        self.assertLess(order.index("config.py"), order.index("model.py"))

        # Model should come before trainer
        self.assertLess(order.index("model.py"), order.index("trainer.py"))

        # Main should be last
        self.assertEqual(order.index("main.py"), len(order) - 1)

    def test_symbol_tracking_across_files(self):
        """Test that symbols are tracked correctly across files."""
        memory = CodeMemory()

        # Add config with class definition
        memory.add_file(
            "config.py",
            '''class Config:
    hidden_size = 256
    learning_rate = 0.001
''',
            purpose="Configuration",
        )

        # Add model with class definition
        memory.add_file(
            "model.py",
            '''class Model:
    def __init__(self):
        self.size = 256
''',
            purpose="Model",
        )

        # Verify symbols are indexed
        config_symbol = memory._symbol_index.get_symbol("Config")
        self.assertIsNotNone(config_symbol)
        self.assertEqual(config_symbol.kind, "class")

        model_symbol = memory._symbol_index.get_symbol("Model")
        self.assertIsNotNone(model_symbol)
        self.assertEqual(model_symbol.kind, "class")

    def test_rag_pattern_injection(self):
        """Test that RAG patterns are retrieved and used."""
        kb = CodeKnowledgeBase.from_builtin()

        # Search for training patterns
        patterns = kb.search("pytorch training loop validation", k=3)

        self.assertGreater(len(patterns), 0)

        # Verify patterns have code
        for pattern in patterns:
            self.assertTrue(len(pattern.code) > 0)
            self.assertTrue(len(pattern.tags) > 0)

        # Get pattern context
        context = patterns[0].to_context()
        self.assertIn("Pattern:", context)


class TestReproAgentE2E(unittest.TestCase):
    """End-to-end tests for ReproAgent."""

    def setUp(self):
        self.paper_context = PaperContext(
            title="Simple MLP Classifier",
            abstract="We propose a simple MLP for classification.",
            method_section="The model consists of fully connected layers.",
        )

    def test_legacy_mode_runs(self):
        """Test that legacy mode still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = ReproAgent({
                "use_orchestrator": False,
                "timeout_sec": 10,
            })

            # Mock the nodes to avoid LLM calls
            agent.planning_node.run = AsyncMock(return_value=MagicMock(
                success=True,
                data=MagicMock(
                    file_structure={"main.py": "entry"},
                    entry_point="main.py",
                    dependencies=[],
                    key_components=[],
                ),
            ))

            agent.environment_node.run = AsyncMock(return_value=MagicMock(
                success=True,
                data=MagicMock(
                    python_version="3.10",
                    base_image="python:3.10-slim",
                    pip_requirements=[],
                ),
            ))

            agent.analysis_node.run = AsyncMock(return_value=MagicMock(
                success=True,
                data=MagicMock(
                    model_type="mlp",
                    layers=[],
                    optimizer="adam",
                    extra_params={},
                ),
            ))

            agent.generation_node.run = AsyncMock(return_value=MagicMock(
                success=True,
                data={"main.py": "print('hello')"},
            ))

            agent.verification_node.run = AsyncMock(return_value=MagicMock(
                success=True,
                data=MagicMock(
                    all_passed=True,
                    syntax_ok=True,
                    imports_ok=True,
                    errors=[],
                    repairs_attempted=0,
                    repairs_successful=0,
                    to_dict=lambda: {"all_passed": True},
                ),
            ))

            result = asyncio.run(agent.reproduce_from_paper(
                self.paper_context,
                output_dir=Path(tmpdir),
            ))

            # Should complete
            self.assertIn("planning", result.phases_completed)

    def test_orchestrator_mode_init(self):
        """Test that orchestrator mode initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = ReproAgent({
                "use_orchestrator": True,
                "use_rag": True,
                "max_context_tokens": 4000,
            })

            orchestrator = agent.get_orchestrator(Path(tmpdir))

            self.assertIsNotNone(orchestrator)
            self.assertTrue(orchestrator.config.use_rag)


if __name__ == "__main__":
    unittest.main()
