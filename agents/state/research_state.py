# agents/state/research_state.py
"""
Research-specific state management for PaperBot Agents.
Tracks paragraph-level progress, search history, and reflection iterations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from .base_state import BaseState, StateStatus


@dataclass
class SearchRecord:
    """Record of a single search operation."""
    query: str
    source: str  # "semantic_scholar", "arxiv", etc.
    result_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    has_results: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "source": self.source,
            "result_count": self.result_count,
            "timestamp": self.timestamp.isoformat(),
            "has_results": self.has_results,
            "error": self.error,
        }


@dataclass
class ParagraphState:
    """
    State for a single research paragraph/section.
    Tracks search → summarize → reflect cycle.
    """
    title: str
    content: str = ""
    is_completed: bool = False
    search_history: List[SearchRecord] = field(default_factory=list)
    latest_summary: str = ""
    reflection_count: int = 0
    max_reflections: int = 3
    
    def add_search(self, record: SearchRecord) -> None:
        """Add a search record to history."""
        self.search_history.append(record)
    
    def set_summary(self, summary: str) -> None:
        """Set the latest summary."""
        self.latest_summary = summary
    
    def increment_reflection(self) -> bool:
        """
        Increment reflection count.
        Returns True if more reflections allowed.
        """
        self.reflection_count += 1
        return self.reflection_count < self.max_reflections
    
    def mark_completed(self) -> None:
        """Mark this paragraph as completed."""
        self.is_completed = True
    
    def get_progress(self) -> float:
        """Get paragraph progress (0-1)."""
        if self.is_completed:
            return 1.0
        if not self.latest_summary:
            return 0.3  # Searched but not summarized
        if self.reflection_count == 0:
            return 0.5  # Summarized but not reflected
        return 0.5 + (0.5 * self.reflection_count / self.max_reflections)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "is_completed": self.is_completed,
            "search_count": len(self.search_history),
            "reflection_count": self.reflection_count,
            "progress": self.get_progress(),
        }


@dataclass
class ResearchState(BaseState):
    """
    State for research-type agents (ResearchAgent, ReviewerAgent, VerificationAgent).
    
    Tracks:
    - Paragraph-level progress
    - Search history across all paragraphs
    - Reflection iterations
    - Overall research progress
    """
    
    paragraphs: List[ParagraphState] = field(default_factory=list)
    current_paragraph_index: int = 0
    total_searches: int = 0
    
    # ==================== Paragraph Management ====================
    
    def add_paragraph(self, title: str, content: str = "") -> ParagraphState:
        """Add a new paragraph to track."""
        para = ParagraphState(title=title, content=content)
        self.paragraphs.append(para)
        return para
    
    def get_current_paragraph(self) -> Optional[ParagraphState]:
        """Get the current paragraph being processed."""
        if 0 <= self.current_paragraph_index < len(self.paragraphs):
            return self.paragraphs[self.current_paragraph_index]
        return None
    
    def advance_paragraph(self) -> bool:
        """
        Move to next paragraph.
        Returns True if there are more paragraphs, False if all done.
        """
        self.current_paragraph_index += 1
        return self.current_paragraph_index < len(self.paragraphs)
    
    # ==================== Search Tracking ====================
    
    def record_search(self, query: str, source: str, result_count: int, error: str = None) -> None:
        """Record a search operation for the current paragraph."""
        record = SearchRecord(
            query=query,
            source=source,
            result_count=result_count,
            has_results=result_count > 0,
            error=error,
        )
        para = self.get_current_paragraph()
        if para:
            para.add_search(record)
        self.total_searches += 1
    
    # ==================== Progress Calculation ====================
    
    def get_progress(self) -> float:
        """
        Get overall progress as a float between 0 and 1.
        Based on paragraph completion status.
        """
        if self.status == StateStatus.PENDING:
            return 0.0
        if self.status == StateStatus.COMPLETED:
            return 1.0
        if not self.paragraphs:
            return 0.5  # Running but no paragraphs defined
        
        total_progress = sum(p.get_progress() for p in self.paragraphs)
        return total_progress / len(self.paragraphs)
    
    def get_completed_count(self) -> int:
        """Get number of completed paragraphs."""
        return sum(1 for p in self.paragraphs if p.is_completed)
    
    def is_all_paragraphs_completed(self) -> bool:
        """Check if all paragraphs are completed."""
        return all(p.is_completed for p in self.paragraphs)
    
    # ==================== Serialization ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize research state to dictionary."""
        base = super().to_dict()
        base.update({
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "current_paragraph_index": self.current_paragraph_index,
            "total_searches": self.total_searches,
            "completed_paragraphs": self.get_completed_count(),
            "total_paragraphs": len(self.paragraphs),
        })
        return base
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the state."""
        progress = self.get_progress()
        completed = self.get_completed_count()
        total = len(self.paragraphs)
        return (
            f"Status: {self.status.value} | "
            f"Progress: {progress:.0%} | "
            f"Paragraphs: {completed}/{total} | "
            f"Searches: {self.total_searches}"
        )
