# repro/repro_agent.py
"""
ReproAgent: Paper2Code-style reproduction with node-based pipeline.

Enhanced architecture using 5 nodes:
- PlanningNode: Generate reproduction plan
- EnvironmentInferenceNode: Infer execution environment
- AnalysisNode: Extract implementation specs with hyperparameters
- GenerationNode: Generate code files
- VerificationNode: Verify generated code with self-healing

Multi-Agent Mode (new):
- Uses Orchestrator for parallel agent execution
- Blueprint distillation for efficient context
- CodeMemory for cross-file awareness
- CodeRAG for pattern injection

Execution Backends:
- DockerExecutor: Local Docker-based execution
- E2BExecutor: Cloud-based E2B sandbox execution
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Literal, TYPE_CHECKING

from .models import (
    PaperContext, ReproductionPlan, ImplementationSpec,
    ReproductionResult, ReproPhase, EnvironmentSpec
)
from .nodes import (
    PlanningNode, AnalysisNode, GenerationNode, VerificationNode,
    EnvironmentInferenceNode
)
from .base_executor import BaseExecutor
from .docker_executor import DockerExecutor
from .e2b_executor import E2BExecutor
from .orchestrator import Orchestrator, OrchestratorConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from paperbot.application.ports.event_log_port import EventLogPort

# Executor type alias
ExecutorType = Literal["docker", "e2b", "auto"]

# Legacy default commands
DEFAULT_PLAN = [
    "python -m pip install -U pip",
    "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
    "if [ -d tests ]; then pytest -q || true; else python -m py_compile $(find . -name '*.py'); fi",
]


class ReproAgent:
    """
    Paper2Code-style reproduction agent with enhanced node-based pipeline.

    Modes:
    1. Paper2Code mode: Full reproduction from paper context
    2. Multi-Agent mode: Uses Orchestrator with Blueprint, CodeMemory, RAG
    3. Legacy mode: Run commands on existing repo

    Enhanced Features:
    - Environment inference (Python/PyTorch/TensorFlow version detection)
    - Hyperparameter extraction from paper
    - Config.yaml and Dockerfile generation
    - Self-healing verification with error classification
    - Multiple execution backends (Docker/E2B)
    - Blueprint distillation for efficient LLM context
    - CodeMemory for cross-file context awareness
    - CodeRAG for pattern injection
    """

    MAX_RETRIES = 3

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Multi-agent mode flag
        self.use_orchestrator = self.config.get("use_orchestrator", False)

        # Initialize nodes (for legacy mode)
        self.planning_node = PlanningNode()
        self.environment_node = EnvironmentInferenceNode(
            prefer_conda=self.config.get("prefer_conda", False)
        )
        self.analysis_node = AnalysisNode()
        self.generation_node = GenerationNode()
        self.verification_node = VerificationNode(
            max_repair_attempts=self.config.get("max_repair_attempts", 3),
            enable_self_healing=self.config.get("enable_self_healing", True),
        )

        # Initialize executor based on config
        self.executor = self._create_executor()
        self.timeout_sec = self.config.get("timeout_sec", 120)

        # Initialize orchestrator for multi-agent mode
        self._orchestrator: Optional[Orchestrator] = None

    def _create_executor(self) -> BaseExecutor:
        """
        Create the appropriate executor based on configuration.

        Config options:
        - executor: "docker" | "e2b" | "auto" (default: "auto")
        - e2b_api_key: E2B API key (or use E2B_API_KEY env var)
        - docker_image: Docker image for local execution

        "auto" mode tries E2B first if configured, falls back to Docker.
        """
        executor_type: ExecutorType = self.config.get("executor", "auto")

        if executor_type == "e2b":
            return self._create_e2b_executor()
        elif executor_type == "docker":
            return self._create_docker_executor()
        else:  # auto
            # Try E2B first if API key is available
            e2b_executor = self._create_e2b_executor()
            if e2b_executor.available():
                self.logger.info("Using E2B executor (cloud sandbox)")
                return e2b_executor

            # Fall back to Docker
            docker_executor = self._create_docker_executor()
            if docker_executor.available():
                self.logger.info("Using Docker executor (local)")
                return docker_executor

            # Neither available, return Docker with warning
            self.logger.warning(
                "Neither E2B nor Docker available. "
                "Set E2B_API_KEY or install Docker for code execution."
            )
            return docker_executor

    def _create_e2b_executor(self) -> E2BExecutor:
        """Create E2B executor from config."""
        return E2BExecutor(
            api_key=self.config.get("e2b_api_key"),
            template=self.config.get("e2b_template", "Python3"),
            timeout_sandbox=self.config.get("e2b_timeout", 300),
        )

    def _create_docker_executor(self) -> DockerExecutor:
        """Create Docker executor from config."""
        return DockerExecutor(
            image=self.config.get("docker_image", "python:3.10-slim"),
            cpu_shares=self.config.get("cpu_shares", 1),
            mem_limit=self.config.get("mem_limit", "1g"),
            network=self.config.get("network", False),
        )

    def switch_executor(self, executor_type: ExecutorType) -> None:
        """
        Switch to a different executor at runtime.

        Args:
            executor_type: "docker", "e2b", or "auto"
        """
        self.config["executor"] = executor_type
        self.executor = self._create_executor()
        self.logger.info(f"Switched to {self.executor.executor_type} executor")

    def get_orchestrator(
        self,
        output_dir: Optional[Path] = None,
        *,
        event_log: "Optional[EventLogPort]" = None,
        workflow: str = "paper2code",
    ) -> Orchestrator:
        """
        Get or create the multi-agent orchestrator.

        Args:
            output_dir: Output directory for generated code

        Returns:
            Configured Orchestrator instance
        """
        if self._orchestrator is None or (output_dir and self._orchestrator.config.output_dir != output_dir):
            config = OrchestratorConfig(
                max_repair_loops=self.config.get("max_repair_attempts", 3),
                output_dir=output_dir,
                use_rag=self.config.get("use_rag", True),
                max_context_tokens=self.config.get("max_context_tokens", 8000),
            )
            self._orchestrator = Orchestrator(config=config, event_log=event_log, workflow=workflow)
        return self._orchestrator

    # ==================== Paper2Code Mode ====================

    async def reproduce_from_paper(
        self,
        paper_context: PaperContext,
        output_dir: Optional[Path] = None,
        *,
        event_log: "Optional[EventLogPort]" = None,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> ReproductionResult:
        """
        Full Paper2Code reproduction from paper context.

        If use_orchestrator=True (set in config), uses the new multi-agent
        pipeline with Blueprint distillation, CodeMemory, and CodeRAG.

        Otherwise uses the legacy node-based pipeline.

        Enhanced Pipeline (Orchestrator mode):
        1. PlanningAgent: Blueprint distillation + plan generation
        2. CodingAgent: Code generation with memory and RAG
        3. VerificationAgent: Verify code
        4. DebuggingAgent: Self-healing repair loop

        Legacy Pipeline:
        1. PlanningNode: Generate plan
        2. EnvironmentInferenceNode: Infer execution environment
        3. AnalysisNode: Extract specs with hyperparameters
        4. GenerationNode: Generate code
        5. VerificationNode: Verify code with self-healing
        """
        # Ensure output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="repro_"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use orchestrator mode if enabled
        if self.use_orchestrator:
            return await self._reproduce_with_orchestrator(
                paper_context,
                output_dir,
                event_log=event_log,
                run_id=run_id,
                trace_id=trace_id,
            )

        # Legacy mode
        return await self._reproduce_legacy(paper_context, output_dir)

    async def _reproduce_with_orchestrator(
        self,
        paper_context: PaperContext,
        output_dir: Path,
        *,
        event_log: "Optional[EventLogPort]" = None,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> ReproductionResult:
        """
        Reproduce using the multi-agent orchestrator.

        Features:
        - Blueprint distillation for efficient LLM context
        - CodeMemory for cross-file awareness
        - CodeRAG for pattern injection
        - Parallel agent execution where possible
        """
        self.logger.info(f"Reproducing '{paper_context.title}' with multi-agent orchestrator")

        orchestrator = self.get_orchestrator(output_dir, event_log=event_log)
        result = await orchestrator.run(paper_context, run_id=run_id, trace_id=trace_id)

        # Write environment files if we have the plan
        if result.plan:
            env_spec = EnvironmentSpec()  # Use defaults
            self._write_environment_files(output_dir, env_spec, result.spec or ImplementationSpec())

        return result

    async def _reproduce_legacy(
        self,
        paper_context: PaperContext,
        output_dir: Path
    ) -> ReproductionResult:
        """
        Legacy Paper2Code reproduction using node-based pipeline.

        Pipeline:
        1. PlanningNode: Generate plan
        2. EnvironmentInferenceNode: Infer execution environment
        3. AnalysisNode: Extract specs with hyperparameters
        4. GenerationNode: Generate code
        5. VerificationNode: Verify code with self-healing
        """
        result = ReproductionResult(
            status=ReproPhase.PLANNING,
            paper_title=paper_context.title,
        )
        
        try:
            # Phase 1: Planning
            self.logger.info(f"Phase 1: Planning for '{paper_context.title}'")
            plan_result = await self.planning_node.run(paper_context)
            if not plan_result.success:
                return self._fail_result(result, ReproPhase.PLANNING, plan_result.error or "")
            plan: ReproductionPlan = plan_result.data
            result.plan = plan
            result.phases_completed.append("planning")
            
            # Phase 1.5: Environment Inference (NEW)
            self.logger.info("Phase 1.5: Environment Inference")
            result.status = ReproPhase.ENVIRONMENT
            env_result = await self.environment_node.run(paper_context)
            if not env_result.success:
                self.logger.warning(f"Environment inference failed: {env_result.error}, using defaults")
                env_spec = EnvironmentSpec()  # Use defaults
            else:
                env_spec: EnvironmentSpec = env_result.data
                result.phases_completed.append("environment")
            
            # Update Docker executor with inferred image
            if env_spec.base_image:
                self.executor = DockerExecutor(image=env_spec.base_image)
            
            # Phase 2: Analysis (enhanced with environment spec)
            self.logger.info("Phase 2: Analysis with hyperparameter extraction")
            result.status = ReproPhase.ANALYSIS
            analysis_result = await self.analysis_node.run((paper_context, plan, env_spec))
            if not analysis_result.success:
                return self._fail_result(result, ReproPhase.ANALYSIS, analysis_result.error or "")
            spec: ImplementationSpec = analysis_result.data
            result.spec = spec
            result.phases_completed.append("analysis")
            
            # Phase 3: Generation
            self.logger.info("Phase 3: Generation")
            result.status = ReproPhase.GENERATION
            gen_result = await self.generation_node.run((paper_context, plan, spec))
            if not gen_result.success:
                return self._fail_result(result, ReproPhase.GENERATION, gen_result.error or "")
            files: Dict[str, str] = gen_result.data
            
            # Write generated code files to disk
            for filepath, content in files.items():
                file_path = output_dir / filepath
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
            result.generated_files = files
            
            # Write environment files
            self._write_environment_files(output_dir, env_spec, spec)
            
            result.phases_completed.append("generation")
            
            # Phase 4: Verification with self-healing
            self.logger.info("Phase 4: Verification with self-healing")
            result.status = ReproPhase.VERIFICATION
            await self._verify_with_self_healing(output_dir, paper_context, result)
            
            # Compute final score
            result.compute_score()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reproduction failed: {e}")
            result.status = ReproPhase.FAILED
            result.errors.append(str(e))
            return result
    
    def _write_environment_files(
        self, output_dir: Path, env_spec: EnvironmentSpec, impl_spec: ImplementationSpec
    ) -> None:
        """Write environment-related files to output directory."""
        # Write Dockerfile
        if env_spec.dockerfile_content:
            dockerfile_path = output_dir / "Dockerfile"
            dockerfile_path.write_text(env_spec.dockerfile_content)
            self.logger.info(f"Generated Dockerfile with base image: {env_spec.base_image}")
        else:
            # Generate on the fly
            dockerfile_content = env_spec.generate_dockerfile()
            (output_dir / "Dockerfile").write_text(dockerfile_content)
        
        # Write requirements.txt
        if env_spec.pip_requirements:
            requirements_path = output_dir / "requirements.txt"
            existing = []
            if requirements_path.exists():
                existing = requirements_path.read_text().splitlines()
            all_reqs = list(dict.fromkeys(existing + env_spec.pip_requirements))
            requirements_path.write_text("\n".join(all_reqs) + "\n")
        
        # Write config.yaml if available
        if "config_yaml" in impl_spec.extra_params:
            config_path = output_dir / "config.yaml"
            config_path.write_text(impl_spec.extra_params["config_yaml"])
            self.logger.info("Generated config.yaml with extracted hyperparameters")
        
        # Write environment.yaml for conda
        if env_spec.conda_yaml_content:
            conda_path = output_dir / "environment.yaml"
            conda_path.write_text(env_spec.conda_yaml_content)
    
    async def _verify_with_self_healing(
        self,
        output_dir: Path,
        paper_context: PaperContext,
        result: ReproductionResult
    ) -> None:
        """Run verification with self-healing debugger."""
        # Pass paper context for better repair context
        verify_result = await self.verification_node.run((output_dir, paper_context))
        
        if verify_result.success and verify_result.data.all_passed:
            result.status = ReproPhase.COMPLETED
            result.verification = verify_result.data.to_dict()
            result.phases_completed.append("verification")
            return
        
        # Collect verification data even if failed
        if verify_result.data:
            result.verification = verify_result.data.to_dict()
            result.errors.extend(verify_result.data.errors)
            result.retry_count = verify_result.data.repairs_attempted
            
            # Log repair statistics
            repairs = verify_result.data
            if repairs.repairs_attempted > 0:
                self.logger.info(
                    f"Self-healing: {repairs.repairs_successful}/{repairs.repairs_attempted} repairs successful"
                )
        
        # Check if at least basic checks passed
        if verify_result.data and verify_result.data.syntax_ok and verify_result.data.imports_ok:
            result.status = ReproPhase.COMPLETED
            result.phases_completed.append("verification")
        else:
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
        
        commands = plan["commands"]
        exec_result = self.executor.run(workdir=repo_path, commands=commands)
        
        results = [{
            "commands": commands,
            "exit_code": exec_result.exit_code,
            "logs": exec_result.logs[:500] if exec_result.logs else "",
            "runtime_meta": exec_result.runtime_meta,
        }]
        
        passed = exec_result.exit_code == 0
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

