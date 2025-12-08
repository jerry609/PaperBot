# repro/nodes/verification_node.py
"""
Verification Node for Paper2Code pipeline.
Phase 4: Verify generated code with syntax check, imports, and tests.
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from .base_node import BaseNode, NodeResult

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verification steps."""
    syntax_ok: bool = False
    imports_ok: bool = False
    tests_ok: bool = False
    smoke_ok: bool = False
    errors: List[str] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        return self.syntax_ok and self.imports_ok
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "syntax_ok": self.syntax_ok,
            "imports_ok": self.imports_ok,
            "tests_ok": self.tests_ok,
            "smoke_ok": self.smoke_ok,
            "errors": self.errors,
            "all_passed": self.all_passed,
        }


class VerificationNode(BaseNode[VerificationResult]):
    """
    Verify generated code with multiple checks.
    
    Input: Path to generated code directory
    Output: VerificationResult
    """

    def __init__(self, timeout: int = 30, **kwargs):
        super().__init__(node_name="VerificationNode", **kwargs)
        self.timeout = timeout
    
    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is a valid directory path."""
        if not isinstance(input_data, (str, Path)):
            return "Input must be a path to generated code directory"
        path = Path(input_data)
        if not path.exists():
            return f"Directory does not exist: {path}"
        return None
    
    async def _execute(self, input_data: Path, **kwargs) -> VerificationResult:
        """Run verification steps."""
        output_dir = Path(input_data)
        result = VerificationResult()
        
        # Step 1: Syntax check
        syntax_result = self._check_syntax(output_dir)
        result.syntax_ok = syntax_result["passed"]
        if not syntax_result["passed"]:
            result.errors.append(f"Syntax: {syntax_result.get('error', 'Unknown error')}")
        
        # Step 2: Import check (only if syntax passed)
        if result.syntax_ok:
            import_result = self._check_imports(output_dir)
            result.imports_ok = import_result["passed"]
            if not import_result["passed"]:
                result.errors.append(f"Imports: {import_result.get('error', 'Unknown error')}")
        
        # Step 3: Run tests (optional)
        if result.imports_ok:
            test_result = self._run_tests(output_dir)
            result.tests_ok = test_result["passed"]
        
        # Step 4: Smoke run (optional)
        if result.imports_ok:
            smoke_result = self._smoke_run(output_dir)
            result.smoke_ok = smoke_result["passed"]
        
        return result
    
    def _check_syntax(self, output_dir: Path) -> Dict[str, Any]:
        """Check Python syntax of all files."""
        py_files = list(output_dir.glob("**/*.py"))
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    code = f.read()
                compile(code, py_file.name, 'exec')
            except SyntaxError as e:
                return {"passed": False, "error": f"{py_file.name}: {e}"}
        
        return {"passed": True}
    
    def _check_imports(self, output_dir: Path) -> Dict[str, Any]:
        """Check if imports work."""
        # Find main.py or first .py file
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
                return {"passed": False, "error": result.stderr[:200]}
            return {"passed": True}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Import check timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _run_tests(self, output_dir: Path) -> Dict[str, Any]:
        """Run pytest if tests directory exists."""
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
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _smoke_run(self, output_dir: Path) -> Dict[str, Any]:
        """Try to run the main entry point with --help."""
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
            # --help usually returns 0
            return {"passed": result.returncode == 0}
        except Exception as e:
            return {"passed": False, "error": str(e)}
