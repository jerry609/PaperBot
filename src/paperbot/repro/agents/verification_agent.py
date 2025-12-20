# repro/agents/verification_agent.py
"""
Verification Agent for Paper2Code pipeline.

Responsible for:
- Syntax checking generated code
- Import verification
- Unit test execution
- Smoke testing
"""

import subprocess
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentResult, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    """Report of verification results."""
    syntax_ok: bool = False
    imports_ok: bool = False
    tests_ok: bool = False
    smoke_ok: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.syntax_ok and self.imports_ok

    @property
    def fully_passed(self) -> bool:
        return self.syntax_ok and self.imports_ok and self.tests_ok and self.smoke_ok

    def to_dict(self) -> Dict[str, Any]:
        return {
            "syntax_ok": self.syntax_ok,
            "imports_ok": self.imports_ok,
            "tests_ok": self.tests_ok,
            "smoke_ok": self.smoke_ok,
            "all_passed": self.all_passed,
            "fully_passed": self.fully_passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class VerificationAgent(BaseAgent):
    """
    Agent responsible for verifying generated code.

    Pipeline:
    1. Check Python syntax of all files
    2. Verify imports work correctly
    3. Run unit tests if available
    4. Perform smoke test on main entry point

    Context Input:
        - output_dir: Path

    Context Output:
        - verification_report: VerificationReport
        - error: str (if any error for debugging)
    """

    def __init__(
        self,
        timeout: int = 30,
        run_tests: bool = True,
        run_smoke: bool = True,
        **kwargs
    ):
        super().__init__(name="VerificationAgent", **kwargs)
        self.timeout = timeout
        self.run_tests = run_tests
        self.run_smoke = run_smoke

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute verification pipeline."""
        output_dir = context.get("output_dir")
        if not output_dir:
            return AgentResult.failure("Missing output_dir in context")

        output_dir = Path(output_dir)
        if not output_dir.exists():
            return AgentResult.failure(f"Output directory does not exist: {output_dir}")

        report = VerificationReport()

        try:
            # Step 1: Syntax check
            self.log("Checking Python syntax...")
            syntax_result = self._check_syntax(output_dir)
            report.syntax_ok = syntax_result["passed"]
            if not syntax_result["passed"]:
                error_msg = syntax_result.get("error", "Unknown syntax error")
                report.errors.append(f"Syntax: {error_msg}")
                self.log(f"Syntax check failed: {error_msg}")

                # Store error for debugging agent
                context["error"] = error_msg
                context["verification_report"] = report

                return AgentResult.success(
                    data={"report": report},
                    metadata={"stage": "syntax", "passed": False}
                )
            else:
                self.log("Syntax check passed")

            # Step 2: Import check
            self.log("Checking imports...")
            import_result = self._check_imports(output_dir)
            report.imports_ok = import_result["passed"]
            if not import_result["passed"]:
                error_msg = import_result.get("error", "Unknown import error")
                report.errors.append(f"Imports: {error_msg}")
                self.log(f"Import check failed: {error_msg}")

                context["error"] = error_msg
                context["verification_report"] = report

                return AgentResult.success(
                    data={"report": report},
                    metadata={"stage": "imports", "passed": False}
                )
            else:
                self.log("Import check passed")

            # Step 3: Run tests (optional)
            if self.run_tests:
                self.log("Running tests...")
                test_result = self._run_tests(output_dir)
                report.tests_ok = test_result["passed"]
                if not test_result["passed"]:
                    if test_result.get("skipped"):
                        report.warnings.append("No tests found")
                        self.log("No tests found - skipping")
                    else:
                        error_msg = test_result.get("error", "Tests failed")
                        report.warnings.append(f"Tests: {error_msg}")
                        self.log(f"Tests failed: {error_msg}")
                else:
                    self.log("Tests passed")

            # Step 4: Smoke test (optional)
            if self.run_smoke:
                self.log("Running smoke test...")
                smoke_result = self._smoke_run(output_dir)
                report.smoke_ok = smoke_result["passed"]
                if not smoke_result["passed"]:
                    if smoke_result.get("skipped"):
                        report.warnings.append("No main.py found")
                        self.log("No main.py found - skipping smoke test")
                    else:
                        error_msg = smoke_result.get("error", "Smoke test failed")
                        report.warnings.append(f"Smoke: {error_msg}")
                        self.log(f"Smoke test failed: {error_msg}")
                else:
                    self.log("Smoke test passed")

            # Store report
            context["verification_report"] = report
            context["error"] = None  # Clear any previous error

            status = "fully_passed" if report.fully_passed else ("passed" if report.all_passed else "partial")
            self.log(f"Verification complete: {status}")

            return AgentResult.success(
                data={"report": report},
                metadata={
                    "stage": "complete",
                    "passed": report.all_passed,
                    "fully_passed": report.fully_passed,
                }
            )

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return AgentResult.failure(str(e))

    def _check_syntax(self, output_dir: Path) -> Dict[str, Any]:
        """Check Python syntax of all files."""
        py_files = list(output_dir.glob("**/*.py"))

        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    code = f.read()
                compile(code, str(py_file), 'exec')
            except SyntaxError as e:
                return {
                    "passed": False,
                    "error": f'File "{py_file}", line {e.lineno}: {e.msg}',
                    "file": str(py_file),
                    "line": e.lineno,
                }

        return {"passed": True}

    def _check_imports(self, output_dir: Path) -> Dict[str, Any]:
        """Check if imports work."""
        main_file = output_dir / "main.py"
        if not main_file.exists():
            py_files = list(output_dir.glob("*.py"))
            if py_files:
                main_file = py_files[0]
            else:
                return {"passed": True}  # No files to check

        try:
            result = subprocess.run(
                ["python3", "-c", f"import sys; sys.path.insert(0, '{output_dir}'); import {main_file.stem}"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            if result.returncode != 0:
                return {"passed": False, "error": result.stderr[:500]}
            return {"passed": True}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Import check timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _run_tests(self, output_dir: Path) -> Dict[str, Any]:
        """Run pytest if tests exist."""
        tests_dir = output_dir / "tests"
        if not tests_dir.exists():
            return {"passed": True, "skipped": True}

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "-q", str(tests_dir)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=output_dir
            )
            return {"passed": result.returncode == 0}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Test run timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _smoke_run(self, output_dir: Path) -> Dict[str, Any]:
        """Try to run main.py with --help."""
        main_file = output_dir / "main.py"
        if not main_file.exists():
            return {"passed": True, "skipped": True}

        try:
            result = subprocess.run(
                ["python3", str(main_file), "--help"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=output_dir
            )
            return {"passed": result.returncode == 0}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Smoke run timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}
