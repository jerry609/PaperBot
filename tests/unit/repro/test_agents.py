# tests/unit/repro/test_agents.py
"""
Unit tests for repro agents module.
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
from repro.agents.base_agent import BaseAgent, AgentResult, AgentStatus
from repro.agents.verification_agent import VerificationAgent, VerificationReport


class TestAgentStatus(unittest.TestCase):
    """Tests for AgentStatus enum."""

    def test_status_values(self):
        self.assertEqual(AgentStatus.IDLE.value, "idle")
        self.assertEqual(AgentStatus.RUNNING.value, "running")
        self.assertEqual(AgentStatus.COMPLETED.value, "completed")
        self.assertEqual(AgentStatus.FAILED.value, "failed")
        self.assertEqual(AgentStatus.WAITING.value, "waiting")


class TestAgentResult(unittest.TestCase):
    """Tests for AgentResult dataclass."""

    def test_success_result(self):
        result = AgentResult.success(data={"key": "value"}, metadata={"step": 1})

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.data, {"key": "value"})
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata, {"step": 1})

    def test_failure_result(self):
        result = AgentResult.failure("Something went wrong")

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIsNone(result.data)
        self.assertEqual(result.error, "Something went wrong")

    def test_to_dict(self):
        result = AgentResult(
            status=AgentStatus.COMPLETED,
            data={"result": 42},
            error=None,
            duration_seconds=1.5,
            messages=["Step 1 done", "Step 2 done"],
            metadata={"info": "test"},
        )

        d = result.to_dict()

        self.assertEqual(d["status"], "completed")
        # data is converted to string
        self.assertIn("42", d["data"])
        self.assertIsNone(d["error"])
        self.assertEqual(d["duration_seconds"], 1.5)
        self.assertEqual(len(d["messages"]), 2)
        self.assertEqual(d["metadata"]["info"], "test")


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""

    def __init__(self, return_value=None, should_fail=False, **kwargs):
        super().__init__(name="TestAgent", **kwargs)
        self.return_value = return_value
        self.should_fail = should_fail
        self.execute_count = 0

    async def execute(self, context):
        self.execute_count += 1
        if self.should_fail:
            raise ValueError("Test error")
        return AgentResult.success(data=self.return_value)


class TestBaseAgent(unittest.TestCase):
    """Tests for BaseAgent class."""

    def test_agent_creation(self):
        agent = ConcreteAgent(max_retries=3)

        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.max_retries, 3)
        self.assertEqual(agent.status, AgentStatus.IDLE)

    def test_successful_run(self):
        agent = ConcreteAgent(return_value={"answer": 42})

        result = asyncio.run(agent.run({}))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.data, {"answer": 42})
        self.assertEqual(agent.execute_count, 1)

    def test_failed_run(self):
        agent = ConcreteAgent(should_fail=True)

        result = asyncio.run(agent.run({}))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("Test error", result.error)

    def test_status_tracking(self):
        agent = ConcreteAgent()

        self.assertEqual(agent.status, AgentStatus.IDLE)

        # Run should update status
        asyncio.run(agent.run({}))

        self.assertEqual(agent.status, AgentStatus.COMPLETED)

    def test_duration_tracking(self):
        agent = ConcreteAgent()

        result = asyncio.run(agent.run({}))

        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_log_method(self):
        agent = ConcreteAgent()

        agent.log("Test message")

        self.assertEqual(len(agent._messages), 1)
        self.assertIn("Test message", agent._messages[0])


class TestVerificationReport(unittest.TestCase):
    """Tests for VerificationReport dataclass."""

    def test_all_passed_true(self):
        report = VerificationReport(
            syntax_ok=True,
            imports_ok=True,
            tests_ok=False,
            smoke_ok=False,
        )

        self.assertTrue(report.all_passed)
        self.assertFalse(report.fully_passed)

    def test_all_passed_false(self):
        report = VerificationReport(
            syntax_ok=True,
            imports_ok=False,
        )

        self.assertFalse(report.all_passed)

    def test_fully_passed(self):
        report = VerificationReport(
            syntax_ok=True,
            imports_ok=True,
            tests_ok=True,
            smoke_ok=True,
        )

        self.assertTrue(report.fully_passed)

    def test_to_dict(self):
        report = VerificationReport(
            syntax_ok=True,
            imports_ok=True,
            errors=["Error 1"],
            warnings=["Warning 1"],
        )

        d = report.to_dict()

        self.assertTrue(d["syntax_ok"])
        self.assertTrue(d["imports_ok"])
        self.assertTrue(d["all_passed"])
        self.assertFalse(d["fully_passed"])
        self.assertEqual(d["errors"], ["Error 1"])
        self.assertEqual(d["warnings"], ["Warning 1"])


class TestVerificationAgent(unittest.TestCase):
    """Tests for VerificationAgent class."""

    def setUp(self):
        self.agent = VerificationAgent(timeout=10, run_tests=False, run_smoke=False)

    def test_missing_output_dir(self):
        result = asyncio.run(self.agent.run({}))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("output_dir", result.error)

    def test_nonexistent_output_dir(self):
        result = asyncio.run(self.agent.run({"output_dir": "/nonexistent/path"}))

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("does not exist", result.error)

    def test_syntax_check_valid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write valid Python
            (Path(tmpdir) / "test.py").write_text("def hello(): pass")

            result = asyncio.run(self.agent.run({"output_dir": tmpdir}))

            self.assertEqual(result.status, AgentStatus.COMPLETED)
            self.assertTrue(result.data["report"].syntax_ok)

    def test_syntax_check_invalid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write invalid Python
            (Path(tmpdir) / "broken.py").write_text("def broken(")

            result = asyncio.run(self.agent.run({"output_dir": tmpdir}))

            self.assertEqual(result.status, AgentStatus.COMPLETED)
            self.assertFalse(result.data["report"].syntax_ok)
            self.assertGreater(len(result.data["report"].errors), 0)

    def test_import_check(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a simple module
            (Path(tmpdir) / "simple.py").write_text("x = 1")

            result = asyncio.run(self.agent.run({"output_dir": tmpdir}))

            self.assertEqual(result.status, AgentStatus.COMPLETED)
            self.assertTrue(result.data["report"].imports_ok)

    def test_context_updated_with_report(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("x = 1")

            context = {"output_dir": tmpdir}
            asyncio.run(self.agent.run(context))

            self.assertIn("verification_report", context)
            self.assertIsInstance(context["verification_report"], VerificationReport)


if __name__ == "__main__":
    unittest.main()
