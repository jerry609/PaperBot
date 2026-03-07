# repro/nodes/verification_node.py
"""
Verification Node for Paper2Code pipeline.
Phase 4: Verify generated code with syntax check, imports, and tests.

Enhanced with Self-Healing Debugger:
- Error classification (syntax/dependency/logic)
- Traceback + source code feedback to LLM
- Automated repair attempts
"""

import subprocess
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from paperbot.repro.verification_runtime import (
    VerificationRuntimePreparationError,
    prepare_verification_runtime,
)
from .base_node import BaseNode, NodeResult
from ..models import ErrorType, PaperContext

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


@dataclass
class VerificationResult:
    """Result of verification steps."""

    syntax_ok: bool = False
    imports_ok: bool = False
    tests_ok: bool = False
    smoke_ok: bool = False
    errors: List[str] = field(default_factory=list)
    repairs_attempted: int = 0
    repairs_successful: int = 0
    repair_log: List[Dict[str, Any]] = field(default_factory=list)
    runtime: Optional[Dict[str, Any]] = None

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
            "repairs_attempted": self.repairs_attempted,
            "repairs_successful": self.repairs_successful,
            "runtime": self.runtime,
        }


@dataclass
class RepairResult:
    """Result of a self-healing repair attempt."""

    success: bool
    error_type: ErrorType
    original_error: str
    fix_description: str = ""
    modified_files: List[str] = field(default_factory=list)
    new_requirements: List[str] = field(default_factory=list)


class ErrorClassifier:
    """Classify errors for targeted repair strategies."""

    # Patterns for error classification
    SYNTAX_PATTERNS = [
        r"SyntaxError:",
        r"IndentationError:",
        r"TabError:",
    ]

    DEPENDENCY_PATTERNS = [
        r"ModuleNotFoundError:",
        r"No module named",
    ]

    LOGIC_PATTERNS = [
        r"TypeError:",
        r"ValueError:",
        r"AttributeError:",
        r"KeyError:",
        r"IndexError:",
        r"RuntimeError:",
        r"ZeroDivisionError:",
    ]

    @classmethod
    def classify(cls, traceback: str) -> Tuple[ErrorType, str]:
        """
        Classify error type from traceback.

        Returns:
            (ErrorType, extracted_detail)
        """
        for pattern in cls.SYNTAX_PATTERNS:
            if re.search(pattern, traceback):
                match = re.search(r"(?:SyntaxError|IndentationError|TabError): (.+)", traceback)
                detail = match.group(1) if match else "Unknown syntax error"
                return ErrorType.SYNTAX, detail

        for pattern in cls.DEPENDENCY_PATTERNS:
            if re.search(pattern, traceback):
                module_match = re.search(r"No module named '([^']+)'", traceback)
                if module_match:
                    return ErrorType.DEPENDENCY, module_match.group(1)
                return ErrorType.DEPENDENCY, "Unknown dependency"

        import_match = re.search(r"cannot import name '([^']+)'", traceback)
        if import_match:
            return ErrorType.LOGIC, import_match.group(1)

        for pattern in cls.LOGIC_PATTERNS:
            if re.search(pattern, traceback):
                match = re.search(rf"{pattern}\s*(.+)", traceback)
                detail = match.group(1) if match else "Unknown logic error"
                return ErrorType.LOGIC, detail

        return ErrorType.UNKNOWN, traceback[:200]

    @classmethod
    def extract_file_and_line(cls, traceback: str) -> Optional[Tuple[str, int]]:
        """Extract the file and line number where error occurred."""
        match = re.search(r'File "([^"]+)", line (\d+)', traceback)
        if match:
            return match.group(1), int(match.group(2))
        return None


class SelfHealingDebugger:
    """
    Attempt to automatically fix code based on error feedback.

    Strategies:
    - SYNTAX: Parse error and fix using LLM code correction
    - DEPENDENCY: Add missing module to requirements.txt
    - LOGIC: Regenerate problematic function using LLM
    """

    SYNTAX_REPAIR_PROMPT = """Fix this Python syntax error.

Error: {error}
File: {filename}
Line: {line_number}

Original code around the error:
```python
{code_context}
```

Provide ONLY the corrected Python code. Do not include explanations.
"""

    LOGIC_REPAIR_PROMPT = """Fix this Python runtime error.

Error Type: {error_type}
Error Message: {error}
Traceback: {traceback}

File: {filename}
Problematic code:
```python
{source_code}
```

Paper context for reference:
{paper_context}

Provide ONLY the corrected Python code. Do not include explanations.
"""

    # Common package name mappings for dependencies
    PACKAGE_MAPPINGS = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "skimage": "scikit-image",
        "bs4": "beautifulsoup4",
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger("SelfHealingDebugger")

    async def repair(
        self,
        error_type: ErrorType,
        traceback: str,
        paper_context: Optional[PaperContext] = None,
    ) -> RepairResult:
        """
        Attempt to repair code based on error type.

        Args:
            error_type: Classification of the error
            traceback: Full traceback string
            paper_context: Original paper context for reference

        Returns:
            RepairResult with success status and modifications
        """
        result = RepairResult(
            success=False,
            error_type=error_type,
            original_error=traceback[:500],
        )

        try:
            if error_type == ErrorType.SYNTAX:
                return await self._repair_syntax(traceback, result)
            elif error_type == ErrorType.DEPENDENCY:
                return await self._repair_dependency(traceback, result)
            elif error_type == ErrorType.LOGIC:
                return await self._repair_logic(traceback, paper_context, result)
            else:
                result.fix_description = "Unknown error type, cannot auto-repair"
                return result
        except Exception as e:
            self.logger.error(f"Repair attempt failed: {e}")
            result.fix_description = f"Repair failed: {str(e)}"
            return result

    async def _repair_syntax(self, traceback: str, result: RepairResult) -> RepairResult:
        """Fix syntax errors using LLM."""
        file_info = ErrorClassifier.extract_file_and_line(traceback)
        if not file_info:
            result.fix_description = "Could not locate syntax error in file"
            return result

        filename, line_number = file_info
        filepath = Path(filename)

        # Read the source file
        if not filepath.exists():
            filepath = self.output_dir / filepath.name
        if not filepath.exists():
            result.fix_description = f"Source file not found: {filename}"
            return result

        source_lines = filepath.read_text().splitlines()

        # Extract context around the error (5 lines before and after)
        start = max(0, line_number - 6)
        end = min(len(source_lines), line_number + 5)
        code_context = "\n".join(
            f"{i+1}: {line}" for i, line in enumerate(source_lines[start:end], start=start)
        )

        if query and ClaudeAgentOptions:
            try:
                prompt = self.SYNTAX_REPAIR_PROMPT.format(
                    error=traceback.split("\n")[-1] if "\n" in traceback else traceback,
                    filename=filepath.name,
                    line_number=line_number,
                    code_context=code_context,
                )

                llm_result = query(prompt=prompt, options=ClaudeAgentOptions(max_tokens=2000))

                fixed_code = self._extract_code_from_response(llm_result.response)
                if fixed_code:
                    # Replace the problematic section
                    new_lines = source_lines[:start] + fixed_code.splitlines() + source_lines[end:]
                    filepath.write_text("\n".join(new_lines))

                    result.success = True
                    result.fix_description = f"Fixed syntax error at line {line_number}"
                    result.modified_files.append(str(filepath))
                    return result
            except Exception as e:
                self.logger.warning(f"LLM syntax repair failed: {e}")

        result.fix_description = "Could not auto-fix syntax error"
        return result

    async def _repair_dependency(self, traceback: str, result: RepairResult) -> RepairResult:
        """Fix missing dependencies by updating requirements.txt."""
        _, missing_module = ErrorClassifier.classify(traceback)

        # Map module name to pip package name
        package_name = self.PACKAGE_MAPPINGS.get(missing_module, missing_module)

        # Handle submodule imports (e.g., transformers.models -> transformers)
        if "." in package_name:
            package_name = package_name.split(".")[0]

        if not package_name or package_name.lower() == "unknown dependency":
            result.fix_description = (
                "Dependency name was not specific enough to update requirements"
            )
            return result
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", package_name):
            result.fix_description = f"Refused to add invalid package name: {package_name}"
            return result

        requirements_file = self.output_dir / "requirements.txt"

        # Read existing requirements
        existing = []
        if requirements_file.exists():
            existing = requirements_file.read_text().splitlines()

        # Check if already in requirements
        if any(package_name in req for req in existing):
            result.fix_description = f"Package {package_name} already in requirements"
            return result

        # Add new requirement
        existing.append(package_name)
        requirements_file.write_text("\n".join(existing) + "\n")

        result.success = True
        result.fix_description = f"Added {package_name} to requirements.txt"
        result.modified_files.append(str(requirements_file))
        result.new_requirements.append(package_name)

        return result

    async def _repair_logic(
        self, traceback: str, paper_context: Optional[PaperContext], result: RepairResult
    ) -> RepairResult:
        """Fix logic errors using LLM."""
        file_info = ErrorClassifier.extract_file_and_line(traceback)
        if not file_info:
            result.fix_description = "Could not locate logic error in file"
            return result

        filename, line_number = file_info
        filepath = Path(filename)

        if not filepath.exists():
            filepath = self.output_dir / filepath.name
        if not filepath.exists():
            result.fix_description = f"Source file not found: {filename}"
            return result

        source_code = filepath.read_text()

        if query and ClaudeAgentOptions:
            try:
                error_type, error_msg = ErrorClassifier.classify(traceback)

                prompt = self.LOGIC_REPAIR_PROMPT.format(
                    error_type=error_type.value,
                    error=error_msg,
                    traceback=traceback[:1000],
                    filename=filepath.name,
                    source_code=source_code[:3000],
                    paper_context=(
                        paper_context.to_prompt_context()[:1000]
                        if paper_context
                        else "Not available"
                    ),
                )

                llm_result = query(prompt=prompt, options=ClaudeAgentOptions(max_tokens=3000))

                fixed_code = self._extract_code_from_response(llm_result.response)
                if fixed_code:
                    filepath.write_text(fixed_code)

                    result.success = True
                    result.fix_description = f"Fixed logic error in {filepath.name}"
                    result.modified_files.append(str(filepath))
                    return result
            except Exception as e:
                self.logger.warning(f"LLM logic repair failed: {e}")

        result.fix_description = "Could not auto-fix logic error"
        return result

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find code block
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find any code block
        code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # If no code block, assume entire response is code
        if response.strip().startswith(("import ", "from ", "def ", "class ")):
            return response.strip()

        return None


class VerificationNode(BaseNode[VerificationResult]):
    """
    Verify generated code with multiple checks and self-healing.

    Enhanced Features:
    - Error classification (syntax/dependency/logic)
    - Self-healing debugger with LLM-based repair
    - Iterative repair attempts

    Input: Path to generated code directory (or tuple with PaperContext)
    Output: VerificationResult
    """

    def __init__(
        self,
        timeout: int = 30,
        max_repair_attempts: int = 3,
        enable_self_healing: bool = True,
        experience_store=None,
        python_executable: str = "python3",
        prepare_requirements: bool = False,
        runtime_cache_dir: Optional[Path] = None,
        install_timeout: int = 600,
        prefer_cpu_torch: bool = False,
        **kwargs,
    ):
        super().__init__(node_name="VerificationNode", **kwargs)
        self.timeout = timeout
        self.max_repair_attempts = max_repair_attempts
        self.enable_self_healing = enable_self_healing
        self._experience_store = experience_store
        self.python_executable = python_executable
        self.prepare_requirements = prepare_requirements
        self.runtime_cache_dir = Path(runtime_cache_dir) if runtime_cache_dir else None
        self.install_timeout = install_timeout
        self.prefer_cpu_torch = prefer_cpu_torch

    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is a valid directory path or tuple."""
        if isinstance(input_data, tuple):
            if len(input_data) < 1:
                return "Tuple must contain at least the output directory path"
            path = Path(input_data[0])
        elif isinstance(input_data, (str, Path)):
            path = Path(input_data)
        else:
            return "Input must be a path or (path, paper_context) tuple"

        if not path.exists():
            return f"Directory does not exist: {path}"
        return None

    async def _execute(self, input_data: Any, **kwargs) -> VerificationResult:
        """Run verification steps with self-healing."""
        # Parse input
        if isinstance(input_data, tuple):
            output_dir = Path(input_data[0])
            paper_context = input_data[1] if len(input_data) > 1 else None
        else:
            output_dir = Path(input_data)
            paper_context = None
        user_id = (kwargs.get("user_id", "default") or "default").strip() or "default"

        result = VerificationResult()
        debugger = SelfHealingDebugger(output_dir) if self.enable_self_healing else None

        # Run verification with repair loop
        for attempt in range(self.max_repair_attempts + 1):
            # Step 1: Syntax check
            syntax_result = self._check_syntax(output_dir)
            result.syntax_ok = syntax_result["passed"]

            if not syntax_result["passed"]:
                error_msg = syntax_result.get("error", "Unknown error")
                result.errors.append(f"Syntax: {error_msg}")

                if debugger and attempt < self.max_repair_attempts:
                    repair_result = await debugger.repair(
                        ErrorType.SYNTAX, error_msg, paper_context
                    )
                    result.repairs_attempted += 1
                    result.repair_log.append(repair_result.__dict__)

                    if repair_result.success:
                        result.repairs_successful += 1
                        self.logger.info(f"Repair successful: {repair_result.fix_description}")
                        continue  # Retry verification
                break

            try:
                runtime = self._prepare_runtime(output_dir)
                result.runtime = runtime.to_dict()
            except VerificationRuntimePreparationError as e:
                result.runtime = e.to_dict()
                result.errors.append(f"Runtime: {e}")
                break
            except Exception as e:
                result.errors.append(f"Runtime: {e}")
                break

            # Step 2: Import check
            import_result = self._check_imports(
                output_dir, python_executable=runtime.python_executable
            )
            result.imports_ok = import_result["passed"]

            if not import_result["passed"]:
                error_msg = import_result.get("error", "Unknown error")
                result.errors.append(f"Imports: {error_msg}")

                error_type, _ = ErrorClassifier.classify(error_msg)

                if debugger and attempt < self.max_repair_attempts:
                    repair_result = await debugger.repair(error_type, error_msg, paper_context)
                    result.repairs_attempted += 1
                    result.repair_log.append(repair_result.__dict__)

                    if repair_result.success:
                        result.repairs_successful += 1
                        self.logger.info(f"Repair successful: {repair_result.fix_description}")
                        continue  # Retry verification
                break

            # If syntax and imports pass, we're done with essential checks
            break

        # Step 3: Run tests (optional, no repair)
        if result.imports_ok:
            test_result = self._run_tests(output_dir, python_executable=runtime.python_executable)
            result.tests_ok = test_result["passed"]

        # Step 4: Smoke run (optional, no repair)
        if result.imports_ok:
            smoke_result = self._smoke_run(output_dir, python_executable=runtime.python_executable)
            result.smoke_ok = smoke_result["passed"]

        # Persist verified structure when all essential checks pass (issue #162)
        if result.all_passed and self._experience_store:
            paper_context = (
                input_data[1] if isinstance(input_data, tuple) and len(input_data) > 1 else None
            )
            paper_id = (
                getattr(paper_context, "paper_id", None) or getattr(paper_context, "arxiv_id", None)
                if paper_context
                else None
            )
            try:
                py_files = [f.name for f in output_dir.glob("*.py")]
                self._experience_store.add(
                    user_id=user_id,
                    pattern_type="verified_structure",
                    content=f"Verified structure in {output_dir.name}: {', '.join(py_files[:10])}",
                    paper_id=paper_id,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to persist verified structure experience.",
                    exc_info=True,
                )

        return result

    def _prepare_runtime(self, output_dir: Path):
        extra_packages = ["pytest"] if (output_dir / "tests").exists() else []
        return prepare_verification_runtime(
            output_dir,
            base_python=self.python_executable,
            prepare_requirements=self.prepare_requirements,
            cache_dir=self.runtime_cache_dir,
            install_timeout=self.install_timeout,
            extra_packages=extra_packages,
            prefer_cpu_torch=self.prefer_cpu_torch,
        )

    def _check_syntax(self, output_dir: Path) -> Dict[str, Any]:
        """Check Python syntax of all files."""
        py_files = list(output_dir.glob("**/*.py"))

        for py_file in py_files:
            try:
                with open(py_file, "r") as f:
                    code = f.read()
                compile(code, str(py_file), "exec")
            except SyntaxError as e:
                return {
                    "passed": False,
                    "error": f'File "{py_file}", line {e.lineno}: {e.msg}',
                    "file": str(py_file),
                    "line": e.lineno,
                }

        return {"passed": True}

    def _check_imports(
        self, output_dir: Path, *, python_executable: str = "python3"
    ) -> Dict[str, Any]:
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
                [
                    python_executable,
                    "-c",
                    f"import sys; sys.path.insert(0, '{output_dir}'); import {main_file.stem}",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                return {"passed": False, "error": result.stderr[:500]}
            return {"passed": True}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Import check timed out"}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _run_tests(self, output_dir: Path, *, python_executable: str = "python3") -> Dict[str, Any]:
        """Run pytest if tests directory exists."""
        tests_dir = output_dir / "tests"
        if not tests_dir.exists():
            return {"passed": True, "skipped": True}

        try:
            result = subprocess.run(
                [python_executable, "-m", "pytest", "-q", str(tests_dir)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=output_dir,
            )
            return {"passed": result.returncode == 0}
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _smoke_run(self, output_dir: Path, *, python_executable: str = "python3") -> Dict[str, Any]:
        """Try to run the main entry point with --help."""
        main_file = output_dir / "main.py"
        if not main_file.exists():
            return {"passed": True, "skipped": True}

        try:
            result = subprocess.run(
                [python_executable, str(main_file), "--help"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=output_dir,
            )
            # --help usually returns 0
            return {"passed": result.returncode == 0}
        except Exception as e:
            return {"passed": False, "error": str(e)}
