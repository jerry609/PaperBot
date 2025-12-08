# agents/state/base_state.py
"""
Base state class for PaperBot Agents.
Provides common state management functionality.
"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import json


class StateStatus(str, Enum):
    """Status enum for agent states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BaseState(ABC):
    """
    Base state class for all agents.
    
    Provides:
    - Status tracking (pending → running → completed/failed)
    - Timing information
    - Error handling
    - Serialization support
    """
    
    status: StateStatus = StateStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ==================== State Transitions ====================
    
    def mark_running(self) -> None:
        """Transition to running state."""
        if self.status not in (StateStatus.PENDING, StateStatus.FAILED):
            raise ValueError(f"Cannot start from status: {self.status}")
        self.status = StateStatus.RUNNING
        self.started_at = datetime.now()
        self.error = None
    
    def mark_completed(self) -> None:
        """Transition to completed state."""
        if self.status != StateStatus.RUNNING:
            raise ValueError(f"Cannot complete from status: {self.status}")
        self.status = StateStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def mark_failed(self, error: str) -> None:
        """Transition to failed state with error message."""
        self.status = StateStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
    
    def mark_cancelled(self) -> None:
        """Transition to cancelled state."""
        self.status = StateStatus.CANCELLED
        self.completed_at = datetime.now()
    
    # ==================== Query Methods ====================
    
    def is_running(self) -> bool:
        """Check if currently running."""
        return self.status == StateStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if completed (success or failure)."""
        return self.status in (StateStatus.COMPLETED, StateStatus.FAILED, StateStatus.CANCELLED)
    
    def is_success(self) -> bool:
        """Check if completed successfully."""
        return self.status == StateStatus.COMPLETED
    
    def get_duration(self) -> Optional[float]:
        """Get duration in seconds, or None if not started."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def get_progress(self) -> float:
        """
        Get progress as a float between 0 and 1.
        Override in subclasses for granular progress.
        """
        if self.status == StateStatus.PENDING:
            return 0.0
        elif self.status == StateStatus.COMPLETED:
            return 1.0
        elif self.status == StateStatus.FAILED:
            return 0.0
        else:  # RUNNING
            return 0.5  # Default to 50% for running
    
    # ==================== Serialization ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata,
            "progress": self.get_progress(),
            "duration_seconds": self.get_duration(),
        }
    
    def to_json(self) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseState":
        """Deserialize state from dictionary."""
        state = cls()
        state.status = StateStatus(data.get("status", "pending"))
        if data.get("started_at"):
            state.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            state.completed_at = datetime.fromisoformat(data["completed_at"])
        state.error = data.get("error")
        state.metadata = data.get("metadata", {})
        return state
    
    def __repr__(self) -> str:
        duration = self.get_duration()
        duration_str = f", duration={duration:.1f}s" if duration else ""
        return f"{self.__class__.__name__}(status={self.status.value}{duration_str})"
