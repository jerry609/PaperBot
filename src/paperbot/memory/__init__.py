"""
Cross-platform long-term memory for chat assistants.

This package focuses on:
- Normalizing chat logs from different providers (ChatGPT/Gemini/etc.)
- Extracting durable "memories" (profile facts, preferences, goals, constraints)
- Providing prompt-ready context blocks for downstream LLM calls
"""

from .schema import MemoryCandidate, NormalizedMessage
from .extractor import extract_memories, build_memory_context
from .parsers.common import parse_chat_log

__all__ = [
    "MemoryCandidate",
    "NormalizedMessage",
    "extract_memories",
    "build_memory_context",
    "parse_chat_log",
]

