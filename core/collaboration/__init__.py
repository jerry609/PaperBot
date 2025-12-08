# core/collaboration/__init__.py
"""
Agent collaboration module for PaperBot.
Enables multi-agent communication and coordination.
"""

from .coordinator import AgentCoordinator
from .messages import AgentMessage, AgentResult

__all__ = [
    "AgentCoordinator",
    "AgentMessage",
    "AgentResult",
]
