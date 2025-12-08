# repro/__init__.py
"""
Paper2Code-style reproduction pipeline with node-based architecture.

Modules:
- models: Data models (PaperContext, ReproductionPlan, etc.)
- nodes: Node-based processing pipeline
- repro_agent: Main orchestrator
- docker_executor: Docker execution environment
"""

from .repro_agent import ReproAgent
from .models import (
    PaperContext,
    ReproductionPlan,
    ImplementationSpec,
    ReproductionResult,
    VerificationResult,
    VerificationStep,
    ReproPhase,
)
from .nodes import (
    BaseNode,
    NodeResult,
    PlanningNode,
    AnalysisNode,
    GenerationNode,
    VerificationNode,
)
from .docker_executor import DockerExecutor
from .execution_result import ExecutionResult

__all__ = [
    # Main Agent
    "ReproAgent",
    "DockerExecutor",
    # Data Models
    "PaperContext",
    "ReproductionPlan",
    "ImplementationSpec",
    "ReproductionResult",
    "VerificationResult",
    "VerificationStep",
    "ReproPhase",
    # Nodes
    "BaseNode",
    "NodeResult",
    "PlanningNode",
    "AnalysisNode",
    "GenerationNode",
    "VerificationNode",
    # Execution
    "ExecutionResult",
]

