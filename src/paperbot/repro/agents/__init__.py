# repro/agents/__init__.py
"""
Multi-Agent Module for Paper2Code pipeline.

Provides specialized agents for parallel execution:
- PlanningAgent: Blueprint distillation and plan generation
- CodingAgent: Code generation with CodeMemory and RAG
- DebuggingAgent: Error detection and repair
- VerificationAgent: Code verification and testing
"""

from .base_agent import BaseAgent, AgentResult, AgentStatus
from .planning_agent import PlanningAgent
from .coding_agent import CodingAgent
from .debugging_agent import DebuggingAgent
from .verification_agent import VerificationAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AgentStatus",
    "PlanningAgent",
    "CodingAgent",
    "DebuggingAgent",
    "VerificationAgent",
]
