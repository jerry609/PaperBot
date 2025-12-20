# repro/rag/__init__.py
"""
Code RAG (Retrieval-Augmented Generation) module for Paper2Code.

Provides knowledge injection for code generation:
- CodeKnowledgeBase: Tag-based pattern matching
- CodePattern: Reusable code templates
- Built-in patterns for common ML tasks
"""

from .knowledge_base import CodeKnowledgeBase, CodePattern

__all__ = [
    "CodeKnowledgeBase",
    "CodePattern",
]
