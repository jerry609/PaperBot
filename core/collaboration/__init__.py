# core/collaboration/__init__.py
"""
Agent collaboration module for PaperBot.
Enables multi-agent communication and coordination.
"""

from .coordinator import AgentCoordinator
from .messages import AgentMessage, AgentResult, MessageType
from .bus import CollaborationBus
from .host import HostOrchestrator, HostConfig

__all__ = [
    "AgentCoordinator",
    "AgentMessage",
    "AgentResult",
    "MessageType",
    "CollaborationBus",
    "HostOrchestrator",
    "HostConfig",
]
