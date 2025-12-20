# repro/agents/base_agent.py
"""
Base Agent class for multi-agent Paper2Code pipeline.

Provides common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List
import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent execution."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"  # Waiting for dependencies


@dataclass
class AgentResult:
    """Result of an agent execution."""
    status: AgentStatus
    data: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: Any, **kwargs) -> "AgentResult":
        """Create a successful result."""
        return cls(status=AgentStatus.COMPLETED, data=data, **kwargs)

    @classmethod
    def failure(cls, error: str, **kwargs) -> "AgentResult":
        """Create a failed result."""
        return cls(status=AgentStatus.FAILED, error=error, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "data": str(self.data)[:500] if self.data else None,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "messages": self.messages,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Base class for all Paper2Code agents.

    Features:
    - Async execution support
    - Status tracking
    - Message bus for inter-agent communication
    - Retry logic
    """

    def __init__(
        self,
        name: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.name = name or self.__class__.__name__
        self.max_retries = max_retries
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agent.{self.name}")
        self._messages: List[str] = []

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's main task.

        Args:
            context: Shared context with input data and dependencies

        Returns:
            AgentResult with output data
        """
        pass

    async def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        Run the agent with error handling and retry logic.

        Args:
            context: Shared context dictionary

        Returns:
            AgentResult with status and output
        """
        start_time = datetime.now()
        self.status = AgentStatus.RUNNING
        self._messages = []

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self.log(f"Starting execution (attempt {attempt + 1})")
                result = await self.execute(context)

                duration = (datetime.now() - start_time).total_seconds()
                result.duration_seconds = duration
                result.messages = self._messages

                self.status = result.status
                self.log(f"Completed in {duration:.2f}s with status {result.status.value}")
                return result

            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt < self.max_retries:
                    self.log(f"Retrying ({attempt + 2}/{self.max_retries + 1})...")
                    await asyncio.sleep(1)  # Brief delay before retry

        # All attempts failed
        duration = (datetime.now() - start_time).total_seconds()
        self.status = AgentStatus.FAILED
        return AgentResult.failure(
            error=last_error or "Unknown error",
            duration_seconds=duration,
            messages=self._messages,
        )

    def log(self, message: str) -> None:
        """Log a message and store it for reporting."""
        self.logger.info(message)
        self._messages.append(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, status={self.status.value})"
