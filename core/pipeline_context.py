from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class PipelineContext:
    paper_id: str
    paper_title: str
    env_info: str = ""
    data_time: str = ""
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    fallbacks: List[str] = field(default_factory=list)
    status: str = "running"

    def record_timing(self, stage: str, seconds: float):
        self.timings[stage] = seconds

    def add_error(self, err: str):
        self.errors.append(err)

    def add_fallback(self, fb: str):
        self.fallbacks.append(fb)

    def mark_completed(self, status: str = "success"):
        self.status = status
        self.completed_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "env_info": self.env_info,
            "data_time": self.data_time,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "timings": self.timings,
            "errors": self.errors,
            "fallbacks": self.fallbacks,
            "status": self.status,
        }

