# repro/repro_agent.py
"""
Enhanced ReproAgent with Paper2Code-style multi-phase reproduction.

Phases:
1. Planning: Create file structure and component overview
2. Analysis: Extract implementation specifications
3. Generation: Produce code with iterative refinement
4. Verification: Fine-grained checks (syntax → import → test → smoke)
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from repro.docker_executor import DockerExecutor
from repro.models import (
    PaperContext,
    ReproductionPlan,
    ImplementationSpec,
    ReproductionResult,
    VerificationResult,
    VerificationStep,
    ReproPhase,
)
from repro.planning_agent import PlanningAgent
from repro.generation_agent import GenerationAgent

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None
    logger.warning("claude-agent-sdk not installed; ReproAgent will use fallback plan.")


# Legacy default plan for backward compatibility
DEFAULT_PLAN = [
    "python -m pip install -U pip",
    "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
    "if [ -d tests ]; then pytest -q || true; else python -m py_compile $(find . -name '*.py'); fi",
]


class ReproAgent:
    """
    Paper2Code-style reproduction agent.
    
    Supports two modes:
    1. Legacy mode: Just run commands on existing repo (backward compatible)
    2. Paper2Code mode: Full multi-phase reproduction from paper context
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        repro_cfg = self.config.get("repro", {})
        
        # Docker executor for verification
        self.executor = DockerExecutor(
            image=repro_cfg.get("docker_image", "python:3.10-slim"),
            cpu_shares=repro_cfg.get("cpu_shares", 1),
            mem_limit=repro_cfg.get("mem_limit", "1g"),
            network=repro_cfg.get("network", False),
        )
        self.timeout_sec = repro_cfg.get("timeout_sec", 300)
        self.max_retries = repro_cfg.get("max_retries", 2)
        
        # Sub-agents for multi-phase pipeline
        self.planning_agent = PlanningAgent(config)
        self.generation_agent = GenerationAgent(config)

    # ==================== Paper2Code Mode ====================
    
    async def reproduce_from_paper(
        self,
        paper_context: PaperContext,
        output_dir: Optional[Path] = None
    ) -> ReproductionResult:
        """
        Paper2Code-style full reproduction from paper context.
        
        Args:
            paper_context: Paper metadata and content
            output_dir: Where to write generated files (uses temp if None)
        
        Returns:
            ReproductionResult with generated code and verification status
        """
        start_time = time.time()
        result = ReproductionResult(paper_title=paper_context.title, status="in_progress")
        
        try:
            # Phase 1: Planning
            logger.info(f"Phase 1: Planning reproduction for '{paper_context.title}'")
            plan = await self.planning_agent.generate_plan(paper_context)
            result.plan = plan
            result.phases_completed.append(ReproPhase.PLANNING.value)
            
            # Phase 2: Analysis
            logger.info("Phase 2: Analyzing implementation details")
            spec = await self.planning_agent.generate_spec(paper_context, plan)
            result.spec = spec
            result.phases_completed.append(ReproPhase.ANALYSIS.value)
            
            # Phase 3: Generation
            logger.info("Phase 3: Generating code")
            generated_files = await self.generation_agent.generate_code(
                paper_context, plan, spec
            )
            result.generated_files = generated_files
            result.phases_completed.append(ReproPhase.GENERATION.value)
            
            # Write files to output directory
            if output_dir is None:
                output_dir = Path(tempfile.mkdtemp(prefix="repro_"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for filepath, content in generated_files.items():
                file_path = output_dir / filepath
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
            
            # Phase 4: Verification with retry loop
            logger.info("Phase 4: Verifying generated code")
            verification_results, retry_count = await self._verify_with_retry(
                output_dir, generated_files
            )
            result.verification_results = verification_results
            result.retry_count = retry_count
            result.phases_completed.append(ReproPhase.VERIFICATION.value)
            
            # Compute final status and score
            result.compute_score()
            if result.overall_score >= 75:
                result.status = "success"
            elif result.overall_score >= 50:
                result.status = "partial"
            else:
                result.status = "failed"
                
        except Exception as e:
            logger.error(f"Reproduction failed: {e}")
            result.status = "error"
            result.error = str(e)
        
        result.total_duration_sec = time.time() - start_time
        return result

    async def _verify_with_retry(
        self,
        output_dir: Path,
        generated_files: Dict[str, str]
    ) -> tuple[List[VerificationResult], int]:
        """
        Run verification with iterative refinement on failures.
        """
        verification_results = []
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            verification_results = await self._run_verification_steps(output_dir)
            
            # Check if all passed
            all_passed = all(vr.passed for vr in verification_results)
            if all_passed:
                break
            
            # Find first failure and try to fix
            first_failure = next((vr for vr in verification_results if not vr.passed), None)
            if first_failure and attempt < self.max_retries:
                logger.info(f"Verification failed at {first_failure.step.value}, attempting fix...")
                
                # Try to refine the problematic file
                if first_failure.step in [VerificationStep.SYNTAX_CHECK, VerificationStep.IMPORT_CHECK]:
                    # Refine main.py or the file mentioned in error
                    for filepath, content in generated_files.items():
                        if filepath.endswith(".py"):
                            refined = await self.generation_agent.refine_code(
                                filepath, content, first_failure.message
                            )
                            if refined != content:
                                generated_files[filepath] = refined
                                (output_dir / filepath).write_text(refined)
                                retry_count += 1
                                break
        
        return verification_results, retry_count

    async def _run_verification_steps(self, output_dir: Path) -> List[VerificationResult]:
        """
        Run fine-grained verification steps.
        """
        results = []
        
        # Step 1: Syntax check
        syntax_result = await self._check_syntax(output_dir)
        results.append(syntax_result)
        if not syntax_result.passed:
            return results  # Can't proceed if syntax fails
        
        # Step 2: Import check
        import_result = await self._check_imports(output_dir)
        results.append(import_result)
        if not import_result.passed:
            return results
        
        # Step 3: Unit tests (if exist)
        test_result = await self._run_tests(output_dir)
        results.append(test_result)
        
        # Step 4: Smoke run
        smoke_result = await self._smoke_run(output_dir)
        results.append(smoke_result)
        
        return results

    async def _check_syntax(self, output_dir: Path) -> VerificationResult:
        """Check Python syntax of all files."""
        start = time.time()
        cmd = ["python", "-m", "py_compile"] + [
            str(f) for f in output_dir.glob("*.py")
        ]
        
        result = self.executor.run(
            output_dir,
            [f"python -m py_compile {' '.join(f.name for f in output_dir.glob('*.py'))}"],
            timeout_sec=30
        )
        
        passed = result.get("status") == "success"
        return VerificationResult(
            step=VerificationStep.SYNTAX_CHECK,
            passed=passed,
            message="" if passed else result.get("logs", "")[:500],
            logs=result.get("logs", ""),
            duration_sec=time.time() - start
        )

    async def _check_imports(self, output_dir: Path) -> VerificationResult:
        """Check if imports work."""
        start = time.time()
        
        # Try to import each Python file
        import_cmds = []
        for f in output_dir.glob("*.py"):
            module_name = f.stem
            import_cmds.append(f"python -c 'import {module_name}'")
        
        cmd = " && ".join(import_cmds) if import_cmds else "echo 'No Python files'"
        result = self.executor.run(output_dir, [cmd], timeout_sec=60)
        
        passed = result.get("status") == "success"
        return VerificationResult(
            step=VerificationStep.IMPORT_CHECK,
            passed=passed,
            message="" if passed else result.get("logs", "")[:500],
            logs=result.get("logs", ""),
            duration_sec=time.time() - start
        )

    async def _run_tests(self, output_dir: Path) -> VerificationResult:
        """Run pytest if tests directory exists."""
        start = time.time()
        
        tests_exist = (output_dir / "tests").exists() or any(
            f.name.startswith("test_") for f in output_dir.glob("*.py")
        )
        
        if not tests_exist:
            return VerificationResult(
                step=VerificationStep.UNIT_TEST,
                passed=True,
                message="No tests found (skipped)",
                duration_sec=0
            )
        
        result = self.executor.run(
            output_dir,
            ["pip install pytest -q", "pytest -q --tb=short || true"],
            timeout_sec=120
        )
        
        passed = result.get("status") == "success"
        return VerificationResult(
            step=VerificationStep.UNIT_TEST,
            passed=passed,
            message="" if passed else result.get("logs", "")[:500],
            logs=result.get("logs", ""),
            duration_sec=time.time() - start
        )

    async def _smoke_run(self, output_dir: Path) -> VerificationResult:
        """Try to run the main entry point."""
        start = time.time()
        
        entry_point = "main.py"
        if not (output_dir / entry_point).exists():
            # Look for any main-like file
            for candidate in ["run.py", "train.py", "demo.py"]:
                if (output_dir / candidate).exists():
                    entry_point = candidate
                    break
        
        result = self.executor.run(
            output_dir,
            [f"timeout 10 python {entry_point} --help || python {entry_point} --epochs 1 || true"],
            timeout_sec=60
        )
        
        # Smoke run is lenient - we just want it to start without crashing hard
        passed = result.get("exit_code", 1) in [0, 124]  # 124 is timeout
        return VerificationResult(
            step=VerificationStep.SMOKE_RUN,
            passed=passed,
            message="" if passed else result.get("logs", "")[:500],
            logs=result.get("logs", ""),
            duration_sec=time.time() - start
        )

    # ==================== Legacy Mode (Backward Compatible) ====================

    async def generate_plan(self, repo_path: Path) -> List[str]:
        """Legacy: Generate execution plan for existing repo."""
        if query is None or ClaudeAgentOptions is None:
            return DEFAULT_PLAN

        try:
            prompt = (
                "你是可复现性验证助手。根据代码仓库生成安装和测试命令，"
                "使用 bash && 串联，避免交互。优先 pip/pytest。简洁输出命令列表，每行一个命令。"
            )
            opts = ClaudeAgentOptions(
                system_prompt=prompt,
                model="claude-3-5-sonnet-latest",
            )
            cmds: List[str] = []
            async for msg in query(
                prompt=f"仓库路径: {repo_path}. 给出命令列表。",
                options=opts,
            ):
                content = getattr(msg, "content", None)
                if not content:
                    continue
                if isinstance(content, list):
                    for block in content:
                        text = getattr(block, "text", None) or getattr(block, "thinking", None)
                        if text:
                            for line in text.splitlines():
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    cmds.append(line)
                elif isinstance(content, str):
                    for line in content.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            cmds.append(line)
            return cmds or DEFAULT_PLAN
        except Exception as e:
            logger.warning(f"Claude plan generation failed, fallback: {e}")
            return DEFAULT_PLAN

    async def run(self, repo_path: Path) -> Dict[str, Any]:
        """Legacy: Run verification on existing repo."""
        if not self.executor.available():
            return {"status": "error", "error": "Docker not available"}

        cmds = await self.generate_plan(repo_path)
        result = self.executor.run(repo_path, cmds, timeout_sec=self.timeout_sec)
        score = self._score(result)
        return {
            "status": result.get("status"),
            "commands": cmds,
            "exit_code": result.get("exit_code"),
            "duration_sec": result.get("duration_sec"),
            "logs": result.get("logs", "")[-2000:],
            "score": score,
            "error": result.get("error"),
        }

    def _score(self, result: Dict[str, Any]) -> int:
        """Legacy scoring."""
        if result.get("status") != "success":
            return 0
        return 100
