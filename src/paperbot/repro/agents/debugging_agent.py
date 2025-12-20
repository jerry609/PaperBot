# repro/agents/debugging_agent.py
"""
Debugging Agent for Paper2Code pipeline.

Responsible for:
- Error classification (syntax/dependency/logic)
- Automated error repair using LLM
- Traceback analysis and fix suggestions
"""

import logging
import re
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult, AgentStatus
from ..models import ErrorType, PaperContext

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


@dataclass
class RepairAttempt:
    """Record of a repair attempt."""
    error_type: ErrorType
    original_error: str
    fix_applied: str
    success: bool
    modified_files: List[str]


class DebuggingAgent(BaseAgent):
    """
    Agent responsible for error detection and repair.

    Pipeline:
    1. Classify error type from traceback
    2. Apply appropriate repair strategy
    3. Track repair attempts for feedback

    Repair Strategies:
    - SYNTAX: Fix Python syntax using LLM
    - DEPENDENCY: Add missing packages to requirements
    - LOGIC: Regenerate problematic code section

    Context Input:
        - error: str (traceback)
        - output_dir: Path
        - paper_context: PaperContext (optional)
        - generated_files: Dict[str, str]

    Context Output:
        - repair_result: RepairAttempt
        - updated_files: Dict[str, str]
    """

    SYNTAX_REPAIR_PROMPT = """Fix this Python syntax error.

Error: {error}
File: {filename}
Line: {line_number}

Original code around the error:
```python
{code_context}
```

Provide ONLY the corrected Python code for this section. No explanations.
"""

    LOGIC_REPAIR_PROMPT = """Fix this Python runtime error.

Error Type: {error_type}
Error Message: {error}
Traceback: {traceback}

File: {filename}
Current code:
```python
{source_code}
```

{paper_context}

Provide ONLY the corrected full file content. No explanations.
"""

    # Package name mappings for common modules
    PACKAGE_MAPPINGS = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "skimage": "scikit-image",
        "bs4": "beautifulsoup4",
    }

    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        super().__init__(name="DebuggingAgent", **kwargs)
        self.output_dir = output_dir
        self.repair_history: List[RepairAttempt] = []

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute debugging pipeline."""
        error = context.get("error")
        if not error:
            return AgentResult.success(
                data={"message": "No error to debug"},
                metadata={"skipped": True}
            )

        output_dir = context.get("output_dir") or self.output_dir
        if not output_dir:
            return AgentResult.failure("Missing output_dir for debugging")

        output_dir = Path(output_dir)
        paper_context = context.get("paper_context")
        generated_files = context.get("generated_files", {})

        try:
            # Step 1: Classify error
            error_type, error_detail = self._classify_error(error)
            self.log(f"Error classified as {error_type.value}: {error_detail[:100]}")

            # Step 2: Apply repair strategy
            repair_result = await self._repair(
                error_type=error_type,
                traceback=error,
                output_dir=output_dir,
                paper_context=paper_context,
                generated_files=generated_files
            )

            self.repair_history.append(repair_result)

            if repair_result.success:
                self.log(f"Repair successful: {repair_result.fix_applied}")
                context["last_repair"] = repair_result
                return AgentResult.success(
                    data={
                        "repair": repair_result,
                        "modified_files": repair_result.modified_files,
                    },
                    metadata={
                        "error_type": error_type.value,
                        "fix_applied": repair_result.fix_applied,
                    }
                )
            else:
                self.log(f"Repair failed: {repair_result.fix_applied}")
                return AgentResult.failure(
                    f"Could not repair {error_type.value} error: {repair_result.fix_applied}"
                )

        except Exception as e:
            logger.error(f"Debugging failed: {e}")
            return AgentResult.failure(str(e))

    def _classify_error(self, traceback: str) -> Tuple[ErrorType, str]:
        """Classify error type from traceback."""
        # Syntax errors
        syntax_patterns = [
            r'SyntaxError:',
            r'IndentationError:',
            r'TabError:',
        ]
        for pattern in syntax_patterns:
            if re.search(pattern, traceback):
                match = re.search(r'(?:SyntaxError|IndentationError|TabError): (.+)', traceback)
                return ErrorType.SYNTAX, match.group(1) if match else "Unknown syntax error"

        # Dependency errors
        if re.search(r'ModuleNotFoundError:|ImportError:', traceback):
            module_match = re.search(r"No module named '([^']+)'", traceback)
            if module_match:
                return ErrorType.DEPENDENCY, module_match.group(1)
            return ErrorType.DEPENDENCY, "Unknown module"

        # Logic/runtime errors
        logic_patterns = [
            r'TypeError:', r'ValueError:', r'AttributeError:',
            r'KeyError:', r'IndexError:', r'RuntimeError:',
        ]
        for pattern in logic_patterns:
            if re.search(pattern, traceback):
                match = re.search(rf'{pattern}\s*(.+)', traceback)
                return ErrorType.LOGIC, match.group(1) if match else "Unknown logic error"

        return ErrorType.UNKNOWN, traceback[:200]

    async def _repair(
        self,
        error_type: ErrorType,
        traceback: str,
        output_dir: Path,
        paper_context: Optional[PaperContext],
        generated_files: Dict[str, str]
    ) -> RepairAttempt:
        """Apply repair based on error type."""
        if error_type == ErrorType.SYNTAX:
            return await self._repair_syntax(traceback, output_dir, generated_files)
        elif error_type == ErrorType.DEPENDENCY:
            return await self._repair_dependency(traceback, output_dir)
        elif error_type == ErrorType.LOGIC:
            return await self._repair_logic(traceback, output_dir, paper_context, generated_files)
        else:
            return RepairAttempt(
                error_type=error_type,
                original_error=traceback[:500],
                fix_applied="Unknown error type - cannot auto-repair",
                success=False,
                modified_files=[]
            )

    async def _repair_syntax(
        self,
        traceback: str,
        output_dir: Path,
        generated_files: Dict[str, str]
    ) -> RepairAttempt:
        """Repair syntax errors using LLM."""
        file_info = self._extract_file_and_line(traceback)
        if not file_info:
            return RepairAttempt(
                error_type=ErrorType.SYNTAX,
                original_error=traceback[:500],
                fix_applied="Could not locate syntax error",
                success=False,
                modified_files=[]
            )

        filename, line_number = file_info
        filepath = output_dir / Path(filename).name

        if not filepath.exists():
            return RepairAttempt(
                error_type=ErrorType.SYNTAX,
                original_error=traceback[:500],
                fix_applied=f"File not found: {filepath}",
                success=False,
                modified_files=[]
            )

        source_lines = filepath.read_text().splitlines()
        start = max(0, line_number - 6)
        end = min(len(source_lines), line_number + 5)
        code_context = "\n".join(
            f"{i+1}: {line}" for i, line in enumerate(source_lines[start:end], start=start)
        )

        if query and ClaudeAgentOptions:
            try:
                prompt = self.SYNTAX_REPAIR_PROMPT.format(
                    error=traceback.split("\n")[-1],
                    filename=filepath.name,
                    line_number=line_number,
                    code_context=code_context,
                )

                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=2000)
                )

                fixed_code = self._extract_code(result.response)
                if fixed_code:
                    new_lines = source_lines[:start] + fixed_code.splitlines() + source_lines[end:]
                    filepath.write_text("\n".join(new_lines))

                    return RepairAttempt(
                        error_type=ErrorType.SYNTAX,
                        original_error=traceback[:500],
                        fix_applied=f"Fixed syntax error at line {line_number}",
                        success=True,
                        modified_files=[str(filepath)]
                    )
            except Exception as e:
                logger.warning(f"LLM syntax repair failed: {e}")

        return RepairAttempt(
            error_type=ErrorType.SYNTAX,
            original_error=traceback[:500],
            fix_applied="Could not auto-fix syntax error",
            success=False,
            modified_files=[]
        )

    async def _repair_dependency(
        self,
        traceback: str,
        output_dir: Path
    ) -> RepairAttempt:
        """Repair missing dependencies."""
        _, missing_module = self._classify_error(traceback)
        package_name = self.PACKAGE_MAPPINGS.get(missing_module, missing_module)

        if "." in package_name:
            package_name = package_name.split(".")[0]

        requirements_file = output_dir / "requirements.txt"
        existing = []
        if requirements_file.exists():
            existing = requirements_file.read_text().splitlines()

        if any(package_name in req for req in existing):
            return RepairAttempt(
                error_type=ErrorType.DEPENDENCY,
                original_error=traceback[:500],
                fix_applied=f"Package {package_name} already in requirements",
                success=False,
                modified_files=[]
            )

        existing.append(package_name)
        requirements_file.write_text("\n".join(existing) + "\n")

        return RepairAttempt(
            error_type=ErrorType.DEPENDENCY,
            original_error=traceback[:500],
            fix_applied=f"Added {package_name} to requirements.txt",
            success=True,
            modified_files=[str(requirements_file)]
        )

    async def _repair_logic(
        self,
        traceback: str,
        output_dir: Path,
        paper_context: Optional[PaperContext],
        generated_files: Dict[str, str]
    ) -> RepairAttempt:
        """Repair logic errors using LLM."""
        file_info = self._extract_file_and_line(traceback)
        if not file_info:
            return RepairAttempt(
                error_type=ErrorType.LOGIC,
                original_error=traceback[:500],
                fix_applied="Could not locate error in file",
                success=False,
                modified_files=[]
            )

        filename, _ = file_info
        filepath = output_dir / Path(filename).name

        if not filepath.exists():
            return RepairAttempt(
                error_type=ErrorType.LOGIC,
                original_error=traceback[:500],
                fix_applied=f"File not found: {filepath}",
                success=False,
                modified_files=[]
            )

        source_code = filepath.read_text()

        if query and ClaudeAgentOptions:
            try:
                error_type, error_msg = self._classify_error(traceback)
                paper_ctx = paper_context.to_prompt_context()[:1000] if paper_context else "Not available"

                prompt = self.LOGIC_REPAIR_PROMPT.format(
                    error_type=error_type.value,
                    error=error_msg,
                    traceback=traceback[:1000],
                    filename=filepath.name,
                    source_code=source_code[:3000],
                    paper_context=f"Paper context: {paper_ctx}" if paper_context else ""
                )

                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=3000)
                )

                fixed_code = self._extract_code(result.response)
                if fixed_code:
                    filepath.write_text(fixed_code)
                    return RepairAttempt(
                        error_type=ErrorType.LOGIC,
                        original_error=traceback[:500],
                        fix_applied=f"Fixed logic error in {filepath.name}",
                        success=True,
                        modified_files=[str(filepath)]
                    )
            except Exception as e:
                logger.warning(f"LLM logic repair failed: {e}")

        return RepairAttempt(
            error_type=ErrorType.LOGIC,
            original_error=traceback[:500],
            fix_applied="Could not auto-fix logic error",
            success=False,
            modified_files=[]
        )

    def _extract_file_and_line(self, traceback: str) -> Optional[Tuple[str, int]]:
        """Extract file and line number from traceback."""
        match = re.search(r'File "([^"]+)", line (\d+)', traceback)
        if match:
            return match.group(1), int(match.group(2))
        return None

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        if response.strip().startswith(("import ", "from ", "def ", "class ")):
            return response.strip()

        return None
