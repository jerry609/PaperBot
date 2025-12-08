# src/paperbot/agents/mixins/__init__.py
"""
Mixins for PaperBot Agents.
Provides reusable functionality following DRY principle.
"""

from .semantic_scholar import SemanticScholarMixin
from .text_parsing import TextParsingMixin
from .json_parser import JSONParserMixin, JSONParseError

__all__ = [
    "SemanticScholarMixin",
    "TextParsingMixin",
    "JSONParserMixin",
    "JSONParseError",
]
