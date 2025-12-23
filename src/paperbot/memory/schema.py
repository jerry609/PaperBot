from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


Role = Literal["system", "user", "assistant", "tool", "unknown"]


@dataclass(frozen=True)
class NormalizedMessage:
    role: Role
    content: str
    ts: Optional[datetime] = None
    platform: str = "unknown"
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


MemoryKind = Literal[
    "profile",
    "preference",
    "goal",
    "project",
    "constraint",
    "todo",
    "fact",
]


@dataclass(frozen=True)
class MemoryCandidate:
    kind: MemoryKind
    content: str
    confidence: float = 0.6
    tags: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

