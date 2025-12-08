"""
统一错误模块。
"""

from .errors import (
    ErrorSeverity,
    PaperBotError,
    LLMError,
    APIError,
    ValidationError,
    Result,
)

__all__ = [
    "ErrorSeverity",
    "PaperBotError",
    "LLMError",
    "APIError",
    "ValidationError",
    "Result",
]

