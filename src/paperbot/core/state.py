# paperbot/core/state.py
"""
状态管理兼容层

导出各种状态类以保持向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class TrackingState:
    """追踪状态"""
    scholar_id: str
    scholar_name: str = ""
    last_checked: str = ""
    papers_processed: int = 0
    papers_total: int = 0
    known_paper_ids: List[str] = field(default_factory=list)
    current_stage: str = "idle"  # idle, fetching, analyzing, complete
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self) -> None:
        """标记开始"""
        self.current_stage = "fetching"
        self.last_checked = datetime.now().isoformat()
    
    def mark_analyzing(self, total: int) -> None:
        """标记正在分析"""
        self.current_stage = "analyzing"
        self.papers_total = total
    
    def mark_progress(self, processed: int) -> None:
        """标记进度"""
        self.papers_processed = processed
    
    def mark_complete(self) -> None:
        """标记完成"""
        self.current_stage = "complete"
    
    def mark_error(self, message: str) -> None:
        """标记错误"""
        self.current_stage = "error"
        self.error_message = message
    
    def add_known_papers(self, paper_ids: List[str]) -> None:
        """添加已知论文"""
        existing = set(self.known_paper_ids)
        existing.update(paper_ids)
        self.known_paper_ids = list(existing)


@dataclass
class ScholarState:
    """学者状态"""
    scholar_id: str
    name: str = ""
    affiliation: str = ""
    h_index: int = 0
    citation_count: int = 0
    paper_count: int = 0
    research_interests: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperState:
    """论文状态"""
    paper_id: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    citation_count: int = 0
    abstract: str = ""
    github_url: Optional[str] = None
    has_code: bool = False
    analysis_status: str = "pending"  # pending, analyzing, complete, error
    analysis_result: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceState:
    """影响力状态"""
    paper_id: str
    total_score: float = 0.0
    academic_score: float = 0.0
    engineering_score: float = 0.0
    recommendation: str = "low"
    metrics_breakdown: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


__all__ = [
    "TrackingState",
    "ScholarState",
    "PaperState",
    "InfluenceState",
]

