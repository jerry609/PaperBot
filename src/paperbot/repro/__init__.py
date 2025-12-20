# repro/__init__.py
"""
Paper2Code-style reproduction pipeline with node-based architecture.

Modules:
- models: Data models (PaperContext, ReproductionPlan, EnvironmentSpec, etc.)
- nodes: Node-based processing pipeline
- repro_agent: Main orchestrator with self-healing
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
    EnvironmentSpec,
    ErrorType,
)
from .nodes import (
    BaseNode,
    NodeResult,
    PlanningNode,
    AnalysisNode,
    GenerationNode,
    VerificationNode,
    EnvironmentInferenceNode,
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
    "EnvironmentSpec",
    "ErrorType",
    # Nodes
    "BaseNode",
    "NodeResult",
    "PlanningNode",
    "AnalysisNode",
    "GenerationNode",
    "VerificationNode",
    "EnvironmentInferenceNode",
    # Execution
    "ExecutionResult",
]

