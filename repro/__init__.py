# repro/__init__.py
"""
Paper2Code-style reproduction pipeline.

Modules:
- models: Data models (PaperContext, ReproductionPlan, etc.)
- planning_agent: Planning and Analysis phases
- generation_agent: Code generation with refinement
- repro_agent: Main orchestrator
- docker_executor: Docker execution environment
"""

from repro.repro_agent import ReproAgent
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
from repro.docker_executor import DockerExecutor

__all__ = [
    "ReproAgent",
    "DockerExecutor",
    "PaperContext",
    "ReproductionPlan",
    "ImplementationSpec",
    "ReproductionResult",
    "VerificationResult",
    "VerificationStep",
    "ReproPhase",
    "PlanningAgent",
    "GenerationAgent",
]
