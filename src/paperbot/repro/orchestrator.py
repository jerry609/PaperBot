# repro/orchestrator.py
"""
Orchestrator for Paper2Code multi-agent pipeline.

Provides:
- Parallel agent execution
- Shared context management
- Repair loop coordination
- Progress tracking
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from .agents import (
    BaseAgent,
    AgentResult,
    AgentStatus,
    PlanningAgent,
    CodingAgent,
    DebuggingAgent,
    VerificationAgent,
)
from .models import PaperContext, ReproductionResult, ReproPhase

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    PLANNING = "planning"
    CODING = "coding"
    VERIFICATION = "verification"
    DEBUGGING = "debugging"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_repair_loops: int = 3
    parallel_agents: bool = True
    timeout_seconds: int = 300
    output_dir: Optional[Path] = None
    use_rag: bool = True
    max_context_tokens: int = 8000


@dataclass
class PipelineProgress:
    """Track pipeline progress."""
    current_stage: PipelineStage = PipelineStage.PLANNING
    stages_completed: List[str] = field(default_factory=list)
    repair_loop_count: int = 0
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage.value,
            "stages_completed": self.stages_completed,
            "repair_loop_count": self.repair_loop_count,
            "duration_seconds": self.duration_seconds,
            "agent_results": {k: v.to_dict() for k, v in self.agent_results.items()},
        }


class Orchestrator:
    """
    Multi-Agent Orchestrator for Paper2Code pipeline.

    Coordinates the execution of specialized agents:
    1. PlanningAgent: Blueprint distillation + plan generation
    2. CodingAgent: Code generation with memory and RAG
    3. VerificationAgent: Syntax/import/test verification
    4. DebuggingAgent: Error repair loop

    Usage:
        orchestrator = Orchestrator(config)
        result = await orchestrator.run(paper_context)
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        on_progress: Optional[Callable[[PipelineProgress], None]] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.on_progress = on_progress
        self.progress = PipelineProgress()

        # Initialize agents
        self.planning_agent = PlanningAgent()
        self.coding_agent = CodingAgent(
            output_dir=self.config.output_dir,
            max_context_tokens=self.config.max_context_tokens,
            use_rag=self.config.use_rag,
        )
        self.verification_agent = VerificationAgent()
        self.debugging_agent = DebuggingAgent(output_dir=self.config.output_dir)

        # Shared context
        self.context: Dict[str, Any] = {}

    async def run(self, paper_context: PaperContext) -> ReproductionResult:
        """
        Run the full Paper2Code pipeline.

        Args:
            paper_context: Input paper context

        Returns:
            ReproductionResult with generated code and verification status
        """
        self.progress = PipelineProgress()
        self.progress.start_time = datetime.now()
        self.context = {"paper_context": paper_context}

        result = ReproductionResult(paper_title=paper_context.title)

        try:
            # Stage 1: Planning
            self._update_stage(PipelineStage.PLANNING)
            planning_result = await self._run_planning()

            if planning_result.status != AgentStatus.COMPLETED:
                result.status = ReproPhase.FAILED
                result.error = f"Planning failed: {planning_result.error}"
                return self._finalize_result(result)

            result.plan = self.context.get("plan")
            result.phases_completed.append("planning")

            # Stage 2: Coding
            self._update_stage(PipelineStage.CODING)
            coding_result = await self._run_coding()

            if coding_result.status != AgentStatus.COMPLETED:
                result.status = ReproPhase.FAILED
                result.error = f"Coding failed: {coding_result.error}"
                return self._finalize_result(result)

            result.generated_files = self.context.get("generated_files", {})
            result.spec = self.context.get("spec")
            result.phases_completed.append("generation")

            # Stage 3: Verification + Repair Loop
            for repair_attempt in range(self.config.max_repair_loops + 1):
                self.progress.repair_loop_count = repair_attempt

                # Verify
                self._update_stage(PipelineStage.VERIFICATION)
                verification_result = await self._run_verification()

                report = self.context.get("verification_report")
                if report:
                    result.verification = report.to_dict()

                # Check if verification passed
                if report and report.all_passed:
                    result.phases_completed.append("verification")
                    break

                # Try to repair if we have errors and attempts remaining
                error = self.context.get("error")
                if error and repair_attempt < self.config.max_repair_loops:
                    self._update_stage(PipelineStage.DEBUGGING)
                    debug_result = await self._run_debugging()

                    result.retry_count += 1

                    if debug_result.status != AgentStatus.COMPLETED:
                        logger.warning(f"Repair attempt {repair_attempt + 1} failed")
                        result.errors.append(f"Repair failed: {debug_result.error}")

                    # Update generated files if repairs were made
                    if debug_result.data and debug_result.data.get("modified_files"):
                        # Re-read files from disk
                        if self.config.output_dir:
                            for filepath in result.generated_files:
                                file_path = self.config.output_dir / filepath
                                if file_path.exists():
                                    result.generated_files[filepath] = file_path.read_text()
                else:
                    # No error or no more repair attempts
                    if error:
                        result.errors.append(error)
                    break

            # Determine final status
            if report and report.all_passed:
                result.status = ReproPhase.COMPLETED
            else:
                result.status = ReproPhase.VERIFICATION
                if not result.error:
                    result.error = "Verification did not pass all checks"

            return self._finalize_result(result)

        except asyncio.TimeoutError:
            result.status = ReproPhase.FAILED
            result.error = "Pipeline timed out"
            return self._finalize_result(result)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result.status = ReproPhase.FAILED
            result.error = str(e)
            return self._finalize_result(result)

    async def _run_planning(self) -> AgentResult:
        """Run planning agent."""
        result = await self.planning_agent.run(self.context)
        self.progress.agent_results["planning"] = result
        self._notify_progress()
        return result

    async def _run_coding(self) -> AgentResult:
        """Run coding agent."""
        # Ensure output_dir is set
        if self.config.output_dir:
            self.context["output_dir"] = self.config.output_dir
            self.coding_agent.output_dir = self.config.output_dir

        result = await self.coding_agent.run(self.context)
        self.progress.agent_results["coding"] = result
        self._notify_progress()
        return result

    async def _run_verification(self) -> AgentResult:
        """Run verification agent."""
        if self.config.output_dir:
            self.context["output_dir"] = self.config.output_dir

        result = await self.verification_agent.run(self.context)
        self.progress.agent_results["verification"] = result
        self._notify_progress()
        return result

    async def _run_debugging(self) -> AgentResult:
        """Run debugging agent."""
        if self.config.output_dir:
            self.debugging_agent.output_dir = self.config.output_dir

        result = await self.debugging_agent.run(self.context)
        self.progress.agent_results["debugging"] = result
        self._notify_progress()
        return result

    def _update_stage(self, stage: PipelineStage) -> None:
        """Update current pipeline stage."""
        if self.progress.current_stage != stage:
            self.progress.stages_completed.append(self.progress.current_stage.value)
        self.progress.current_stage = stage
        self._notify_progress()

    def _notify_progress(self) -> None:
        """Notify progress callback if set."""
        if self.on_progress:
            try:
                self.on_progress(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _finalize_result(self, result: ReproductionResult) -> ReproductionResult:
        """Finalize the result with timing info."""
        self.progress.end_time = datetime.now()
        result.total_duration_sec = self.progress.duration_seconds
        result.compute_score()
        return result


class ParallelOrchestrator(Orchestrator):
    """
    Extended orchestrator with parallel execution support.

    Allows running independent agents in parallel for faster execution.
    """

    async def run_parallel(
        self,
        paper_context: PaperContext,
        run_analysis: bool = True,
    ) -> ReproductionResult:
        """
        Run pipeline with parallel execution where possible.

        Currently parallelizes:
        - Blueprint distillation and analysis (if enabled)

        Args:
            paper_context: Input paper context
            run_analysis: Whether to run analysis in parallel with planning

        Returns:
            ReproductionResult
        """
        # For now, use sequential execution
        # Parallel execution can be extended here for independent tasks
        return await self.run(paper_context)
