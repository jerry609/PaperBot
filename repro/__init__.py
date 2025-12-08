# repro/__init__.py
"""
Paper2Code-style reproduction pipeline with node-based architecture.

Modules:
- models: Data models (PaperContext, ReproductionPlan, etc.)
- nodes: Node-based processing pipeline
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
from repro.nodes import (
    BaseNode,
    NodeResult,
    PlanningNode,
    AnalysisNode,
    GenerationNode,
    VerificationNode,
)
from repro.docker_executor import DockerExecutor

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
]
