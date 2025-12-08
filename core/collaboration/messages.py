# core/collaboration/messages.py
"""
Message models for agent collaboration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List


class MessageType(str, Enum):
    """Types of agent messages."""
    INSIGHT = "insight"        # Agent found something interesting
    REQUEST = "request"        # Agent needs help from others
    RESULT = "result"          # Agent completed a task
    ERROR = "error"            # Agent encountered an error
    SYNTHESIS = "synthesis"    # Coordinator synthesized insights


@dataclass
class AgentMessage:
    """
    Message sent between agents.
    
    Attributes:
        sender: Name of the sending agent
        message_type: Type of message
        content: Message content (text or structured data)
        timestamp: When the message was sent
        metadata: Additional context
    """
    sender: str
    message_type: MessageType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "type": self.message_type.value,
            "content": self.content if isinstance(self.content, (str, dict, list)) else str(self.content),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def insight(cls, sender: str, content: str, **metadata) -> "AgentMessage":
        """Create an insight message."""
        return cls(sender=sender, message_type=MessageType.INSIGHT, content=content, metadata=metadata)
    
    @classmethod
    def result(cls, sender: str, content: Any, **metadata) -> "AgentMessage":
        """Create a result message."""
        return cls(sender=sender, message_type=MessageType.RESULT, content=content, metadata=metadata)
    
    @classmethod
    def error(cls, sender: str, error: str, **metadata) -> "AgentMessage":
        """Create an error message."""
        return cls(sender=sender, message_type=MessageType.ERROR, content=error, metadata=metadata)


@dataclass
class AgentResult:
    """
    Result of an agent's work.
    
    Attributes:
        agent_name: Name of the agent
        task_name: Name of the task completed
        success: Whether the task succeeded
        data: Result data
        messages: Messages generated during the task
        duration: Time taken in seconds
    """
    agent_name: str
    task_name: str
    success: bool
    data: Any = None
    messages: List[AgentMessage] = field(default_factory=list)
    duration: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "task": self.task_name,
            "success": self.success,
            "data": self.data if isinstance(self.data, (str, dict, list)) else str(self.data),
            "message_count": len(self.messages),
            "duration": self.duration,
            "error": self.error,
        }
    
    @classmethod
    def ok(cls, agent_name: str, task_name: str, data: Any, duration: float = 0.0) -> "AgentResult":
        """Create a successful result."""
        return cls(agent_name=agent_name, task_name=task_name, success=True, data=data, duration=duration)
    
    @classmethod
    def fail(cls, agent_name: str, task_name: str, error: str, duration: float = 0.0) -> "AgentResult":
        """Create a failed result."""
        return cls(agent_name=agent_name, task_name=task_name, success=False, error=error, duration=duration)
