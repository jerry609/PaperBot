# repro/__init__.py
"""
Paper2Code-style reproduction pipeline with node-based architecture.

Modules:
- models: Data models (PaperContext, ReproductionPlan, Blueprint, etc.)
- nodes: Node-based processing pipeline
- agents: Multi-agent coordination
- orchestrator: Pipeline orchestration
- memory: Stateful code memory for cross-file context
- rag: Code pattern retrieval
- repro_agent: Main orchestrator with self-healing
- base_executor: Abstract executor interface
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
    Blueprint,
    AlgorithmSpec,
)
from .nodes import (
    BaseNode,
    NodeResult,
    PlanningNode,
    AnalysisNode,
    GenerationNode,
    VerificationNode,
    EnvironmentInferenceNode,
    BlueprintDistillationNode,
)
from .agents import (
    BaseAgent,
    AgentResult,
    AgentStatus,
    PlanningAgent,
    CodingAgent,
    DebuggingAgent,
    VerificationAgent,
)
from .orchestrator import Orchestrator, OrchestratorConfig, ParallelOrchestrator
from .memory import CodeMemory, SymbolIndex
from .rag import CodeKnowledgeBase, CodePattern
from .base_executor import BaseExecutor
from .execution_result import ExecutionResult

__all__ = [
    # Main Agent
    "ReproAgent",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "ParallelOrchestrator",
    # Executors
    "BaseExecutor",
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
    "Blueprint",
    "AlgorithmSpec",
    # Nodes
    "BaseNode",
    "NodeResult",
    "PlanningNode",
    "AnalysisNode",
    "GenerationNode",
    "VerificationNode",
    "EnvironmentInferenceNode",
    "BlueprintDistillationNode",
    # Agents
    "BaseAgent",
    "AgentResult",
    "AgentStatus",
    "PlanningAgent",
    "CodingAgent",
    "DebuggingAgent",
    "VerificationAgent",
    # Memory
    "CodeMemory",
    "SymbolIndex",
    # RAG
    "CodeKnowledgeBase",
    "CodePattern",
    # Execution
    "ExecutionResult",
]

