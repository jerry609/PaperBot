from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ExecutionResult:
    status: str
    exit_code: int
    logs: str = ""
    duration_sec: float = 0.0
    error: Optional[str] = None
    runtime_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "success" and self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "logs": self.logs,
            "duration_sec": self.duration_sec,
            "error": self.error,
            "runtime_meta": self.runtime_meta,
        }

