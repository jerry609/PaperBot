"""
PaperBot API Package
FastAPI backend for CLI and Web interfaces
"""

from .main import app
from .streaming import StreamEvent, wrap_generator

__all__ = ["app", "StreamEvent", "wrap_generator"]
