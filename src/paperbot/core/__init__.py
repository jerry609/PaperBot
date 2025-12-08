"""
核心层：抽象、流水线、依赖注入、错误处理。
"""

from .abstractions import Executable, ExecutionResult, ensure_execution_result
from .pipeline import Pipeline, PipelineStage, PipelineResult, StageResult
from .di import Container, inject, bootstrap_dependencies
from .errors import (
    ErrorSeverity,
    PaperBotError,
    LLMError,
    APIError,
    ValidationError,
    Result,
)

__all__ = [
    # 抽象
    "Executable",
    "ExecutionResult",
    "ensure_execution_result",
    # 流水线
    "Pipeline",
    "PipelineStage",
    "PipelineResult",
    "StageResult",
    # 依赖注入
    "Container",
    "inject",
    "bootstrap_dependencies",
    # 错误
    "ErrorSeverity",
    "PaperBotError",
    "LLMError",
    "APIError",
    "ValidationError",
    "Result",
]
