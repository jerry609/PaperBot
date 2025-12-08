# repro/repro_agent.py
"""
ReproAgent: Paper2Code-style reproduction with node-based pipeline.

Simplified architecture using 4 nodes:
- PlanningNode: Generate reproduction plan
- AnalysisNode: Extract implementation specs
- GenerationNode: Generate code files
- VerificationNode: Verify generated code
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .models import PaperContext, ReproductionPlan, ImplementationSpec, ReproductionResult, ReproPhase
from .nodes import PlanningNode, AnalysisNode, GenerationNode, VerificationNode
from .docker_executor import DockerExecutor

logger = logging.getLogger(__name__)

# Legacy default commands
DEFAULT_PLAN = [
    "python -m pip install -U pip",
    "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
    "if [ -d tests ]; then pytest -q || true; else python -m py_compile $(find . -name '*.py'); fi",
]


class ReproAgent:
    """
    Paper2Code-style reproduction agent with node-based pipeline.
    
    Modes:
    1. Paper2Code mode: Full reproduction from paper context
    2. Legacy mode: Run commands on existing repo
    """
    
    MAX_RETRIES = 3
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize nodes
        self.planning_node = PlanningNode()
        self.analysis_node = AnalysisNode()
        self.generation_node = GenerationNode()
        self.verification_node = VerificationNode()
        
        # Initialize Docker executor for legacy mode
        self.executor = DockerExecutor(
            image=self.config.get("docker_image", "python:3.10-slim"),
            timeout=self.config.get("timeout_sec", 120),
        )
    
    # ==================== Paper2Code Mode ====================
    
    async def reproduce_from_paper(
        self,
        paper_context: PaperContext,
        output_dir: Optional[Path] = None
    ) -> ReproductionResult:
        """
        Full Paper2Code reproduction from paper context.
        
        Pipeline:
        1. PlanningNode: Generate plan
        2. AnalysisNode: Extract specs
        3. GenerationNode: Generate code
        4. VerificationNode: Verify code
        """
        result = ReproductionResult(
            status=ReproPhase.PLANNING,
            paper_title=paper_context.title,
        )
        
        # Ensure output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="repro_"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Phase 1: Planning
            self.logger.info(f"Phase 1: Planning for '{paper_context.title}'")
            plan_result = await self.planning_node.run(paper_context)
            if not plan_result.success:
                return self._fail_result(result, ReproPhase.PLANNING, plan_result.error)
            plan: ReproductionPlan = plan_result.data
            
            # Phase 2: Analysis
            self.logger.info("Phase 2: Analysis")
            result.status = ReproPhase.ANALYSIS
            analysis_result = await self.analysis_node.run((paper_context, plan))
            if not analysis_result.success:
                return self._fail_result(result, ReproPhase.ANALYSIS, analysis_result.error)
            spec: ImplementationSpec = analysis_result.data
            
            # Phase 3: Generation
            self.logger.info("Phase 3: Generation")
            result.status = ReproPhase.GENERATION
            gen_result = await self.generation_node.run((paper_context, plan, spec))
            if not gen_result.success:
                return self._fail_result(result, ReproPhase.GENERATION, gen_result.error)
            files: Dict[str, str] = gen_result.data
            
            # Write files to disk
            for filepath, content in files.items():
                file_path = output_dir / filepath
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
            result.generated_files = files
            
            # Phase 4: Verification with retry
            self.logger.info("Phase 4: Verification")
            result.status = ReproPhase.VERIFICATION
            await self._verify_with_retry(output_dir, files, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reproduction failed: {e}")
            result.status = ReproPhase.FAILED
            result.errors.append(str(e))
            return result
    
    async def _verify_with_retry(
        self,
        output_dir: Path,
        files: Dict[str, str],
        result: ReproductionResult
    ) -> None:
        """Run verification with retry on failures."""
        for attempt in range(self.MAX_RETRIES):
            verify_result = await self.verification_node.run(output_dir)
            
            if verify_result.success and verify_result.data.all_passed:
                result.status = ReproPhase.COMPLETED
                result.verification = verify_result.data.to_dict()
                return
            
            # Log errors and retry
            result.retry_count = attempt + 1
            if verify_result.data:
                result.errors.extend(verify_result.data.errors)
            
            if attempt < self.MAX_RETRIES - 1:
                self.logger.info(f"Verification failed, retry {attempt + 2}/{self.MAX_RETRIES}")
                # TODO: Could use GenerationNode.refine_code here
        
        # All retries exhausted
        result.status = ReproPhase.FAILED
    
    def _fail_result(
        self,
        result: ReproductionResult,
        phase: ReproPhase,
        error: str
    ) -> ReproductionResult:
        """Mark result as failed."""
        result.status = ReproPhase.FAILED
        result.errors.append(f"{phase.value}: {error}")
        return result
    
    # ==================== Legacy Mode ====================
    
    async def generate_plan(self, repo_path: Path) -> Dict[str, Any]:
        """Legacy: Generate execution plan for existing repo."""
        return {
            "commands": DEFAULT_PLAN,
            "repo_path": str(repo_path),
        }
    
    async def run(self, repo_path: Path) -> Dict[str, Any]:
        """Legacy: Run verification on existing repo."""
        plan = await self.generate_plan(repo_path)
        
        results = []
        for cmd in plan["commands"]:
            exec_result = self.executor.run(cmd, volume_path=repo_path)
            results.append({
                "command": cmd,
                "exit_code": exec_result.exit_code,
                "stdout": exec_result.stdout[:500] if exec_result.stdout else "",
                "stderr": exec_result.stderr[:500] if exec_result.stderr else "",
            })
            
            if exec_result.exit_code != 0:
                break
        
        passed = all(r["exit_code"] == 0 for r in results)
        return {
            "passed": passed,
            "results": results,
            "score": self._score({"passed": passed, "results": results}),
        }
    
    def _score(self, result: Dict[str, Any]) -> float:
        """Legacy scoring."""
        if result.get("passed"):
            return 1.0
        # Partial score based on commands completed
        results = result.get("results", [])
        if not results:
            return 0.0
        passed = sum(1 for r in results if r["exit_code"] == 0)
        return passed / len(results)
