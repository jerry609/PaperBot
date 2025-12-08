# agents/state/__init__.py
"""
State management for PaperBot Agents.
Inspired by BettaFish's state management pattern.
"""

from .base_state import BaseState, StateStatus
from .research_state import ResearchState, ParagraphState, SearchRecord

__all__ = [
    "BaseState",
    "StateStatus",
    "ResearchState",
    "ParagraphState",
    "SearchRecord",
]
