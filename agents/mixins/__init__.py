# agents/mixins/__init__.py
"""
Mixins for PaperBot Agents.
Provides reusable functionality following DRY principle.
"""

from .semantic_scholar import SemanticScholarMixin
from .text_parsing import TextParsingMixin

__all__ = [
    "SemanticScholarMixin",
    "TextParsingMixin",
]
