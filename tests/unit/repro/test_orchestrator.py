# tests/unit/repro/test_orchestrator.py
"""
Unit tests for Orchestrator module.
"""

import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock external dependencies
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import unittest
import asyncio
from pathlib import Path
from datetime import datetime

from repro.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    PipelineProgress,
    PipelineStage,
    ParallelOrchestrator,
)
from repro.agents.base_agent import AgentResult, AgentStatus
from repro.models import PaperContext, ReproPhase


class TestOrchestratorConfig(unittest.TestCase):
    """Tests for OrchestratorConfig dataclass."""

    def test_default_values(self):
        config = OrchestratorConfig()

        self.assertEqual(config.max_repair_loops, 3)
        self.assertTrue(config.parallel_agents)
        self.assertEqual(config.timeout_seconds, 300)
        self.assertIsNone(config.output_dir)
        self.assertTrue(config.use_rag)
        self.assertEqual(config.max_context_tokens, 8000)

    def test_custom_values(self):
        config = OrchestratorConfig(
            max_repair_loops=5,
            parallel_agents=False,
            output_dir=Path("/tmp/test"),
            use_rag=False,
        )

        self.assertEqual(config.max_repair_loops, 5)
        self.assertFalse(config.parallel_agents)
        self.assertEqual(config.output_dir, Path("/tmp/test"))
        self.assertFalse(config.use_rag)


class TestPipelineStage(unittest.TestCase):
    """Tests for PipelineStage enum."""

    def test_stage_values(self):
        self.assertEqual(PipelineStage.PLANNING.value, "planning")
        self.assertEqual(PipelineStage.CODING.value, "coding")
        self.assertEqual(PipelineStage.VERIFICATION.value, "verification")
        self.assertEqual(PipelineStage.DEBUGGING.value, "debugging")
        self.assertEqual(PipelineStage.COMPLETED.value, "completed")
        self.assertEqual(PipelineStage.FAILED.value, "failed")


class TestPipelineProgress(unittest.TestCase):
    """Tests for PipelineProgress dataclass."""

    def test_default_values(self):
        progress = PipelineProgress()

        self.assertEqual(progress.current_stage, PipelineStage.PLANNING)
        self.assertEqual(progress.stages_completed, [])
        self.assertEqual(progress.repair_loop_count, 0)
        self.assertEqual(progress.agent_results, {})

    def test_duration_not_started(self):
        progress = PipelineProgress()

        self.assertEqual(progress.duration_seconds, 0.0)

    def test_duration_in_progress(self):
        progress = PipelineProgress()
        progress.start_time = datetime.now()

        # Should return positive duration
        self.assertGreaterEqual(progress.duration_seconds, 0.0)

    def test_duration_completed(self):
        progress = PipelineProgress()
        progress.start_time = datetime.now()
        progress.end_time = datetime.now()

        self.assertGreaterEqual(progress.duration_seconds, 0.0)

    def test_to_dict(self):
        progress = PipelineProgress(
            current_stage=PipelineStage.CODING,
            stages_completed=["planning"],
            repair_loop_count=1,
        )
        progress.start_time = datetime.now()

        d = progress.to_dict()

        self.assertEqual(d["current_stage"], "coding")
        self.assertEqual(d["stages_completed"], ["planning"])
        self.assertEqual(d["repair_loop_count"], 1)
        self.assertIn("duration_seconds", d)


class TestOrchestrator(unittest.TestCase):
    """Tests for Orchestrator class."""

    def setUp(self):
        self.config = OrchestratorConfig(max_repair_loops=1)
        self.orchestrator = Orchestrator(config=self.config)
        self.paper_context = PaperContext(
            title="Test Paper",
            abstract="This is a test abstract for testing purposes.",
        )

    def test_initialization(self):
        self.assertIsNotNone(self.orchestrator.planning_agent)
        self.assertIsNotNone(self.orchestrator.coding_agent)
        self.assertIsNotNone(self.orchestrator.verification_agent)
        self.assertIsNotNone(self.orchestrator.debugging_agent)
        self.assertEqual(self.orchestrator.context, {})

    def test_progress_callback(self):
        progress_updates = []

        def on_progress(progress):
            progress_updates.append(progress.current_stage)

        orchestrator = Orchestrator(
            config=self.config,
            on_progress=on_progress,
        )

        # Mock agents to return quickly
        orchestrator.planning_agent.run = AsyncMock(return_value=AgentResult.failure("Skip"))

        asyncio.run(orchestrator.run(self.paper_context))

        # Should have received at least one progress update
        self.assertGreater(len(progress_updates), 0)

    def test_run_planning_failure(self):
        # Mock planning to fail
        self.orchestrator.planning_agent.run = AsyncMock(
            return_value=AgentResult.failure("Planning failed")
        )

        result = asyncio.run(self.orchestrator.run(self.paper_context))

        self.assertEqual(result.status, ReproPhase.FAILED)
        self.assertIn("Planning failed", result.error)

    def test_run_coding_failure(self):
        # Mock planning to succeed
        self.orchestrator.planning_agent.run = AsyncMock(
            return_value=AgentResult.success(data={"plan": {}})
        )
        # Mock coding to fail
        self.orchestrator.coding_agent.run = AsyncMock(
            return_value=AgentResult.failure("Coding failed")
        )

        result = asyncio.run(self.orchestrator.run(self.paper_context))

        self.assertEqual(result.status, ReproPhase.FAILED)
        self.assertIn("Coding failed", result.error)

    def test_context_shared_between_agents(self):
        # Mock planning to add to context
        async def planning_run(context):
            context["plan"] = {"files": ["main.py"]}
            return AgentResult.success(data={"plan": context["plan"]})

        async def coding_run(context):
            # Should have access to plan
            self.assertIn("plan", context)
            return AgentResult.failure("Stop here")

        self.orchestrator.planning_agent.run = planning_run
        self.orchestrator.coding_agent.run = coding_run

        asyncio.run(self.orchestrator.run(self.paper_context))

        # Verify context was shared
        self.assertIn("plan", self.orchestrator.context)

    def test_result_includes_timing(self):
        # Mock all agents to succeed quickly
        self.orchestrator.planning_agent.run = AsyncMock(
            return_value=AgentResult.success(data={})
        )
        self.orchestrator.coding_agent.run = AsyncMock(
            return_value=AgentResult.success(data={"generated_files": {}})
        )

        # Mock verification to succeed
        mock_report = MagicMock()
        mock_report.all_passed = True
        mock_report.to_dict.return_value = {}

        async def verify_run(context):
            context["verification_report"] = mock_report
            return AgentResult.success(data={"report": mock_report})

        self.orchestrator.verification_agent.run = verify_run

        result = asyncio.run(self.orchestrator.run(self.paper_context))

        self.assertIsNotNone(result.total_duration_sec)
        self.assertGreaterEqual(result.total_duration_sec, 0)

    def test_repair_loop_count(self):
        # Mock planning and coding to succeed
        self.orchestrator.planning_agent.run = AsyncMock(
            return_value=AgentResult.success(data={})
        )
        self.orchestrator.coding_agent.run = AsyncMock(
            return_value=AgentResult.success(data={"generated_files": {}})
        )

        # Mock verification to fail
        mock_report = MagicMock()
        mock_report.all_passed = False
        mock_report.to_dict.return_value = {}

        async def verify_run(context):
            context["verification_report"] = mock_report
            context["error"] = "Verification failed"
            return AgentResult.success(data={"report": mock_report})

        self.orchestrator.verification_agent.run = verify_run
        self.orchestrator.debugging_agent.run = AsyncMock(
            return_value=AgentResult.success(data={})
        )

        result = asyncio.run(self.orchestrator.run(self.paper_context))

        # Should have attempted repairs
        self.assertGreater(result.retry_count, 0)


class TestParallelOrchestrator(unittest.TestCase):
    """Tests for ParallelOrchestrator class."""

    def test_inherits_from_orchestrator(self):
        orchestrator = ParallelOrchestrator()

        self.assertIsInstance(orchestrator, Orchestrator)

    def test_run_parallel_exists(self):
        orchestrator = ParallelOrchestrator()

        self.assertTrue(hasattr(orchestrator, "run_parallel"))
        self.assertTrue(callable(orchestrator.run_parallel))


if __name__ == "__main__":
    unittest.main()
