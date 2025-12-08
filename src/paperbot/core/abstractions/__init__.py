"""
核心抽象：统一 Agent/Node 的执行契约与结果结构。
"""

from .executable import Executable, ExecutionResult, ensure_execution_result

__all__ = [
    "Executable",
    "ExecutionResult",
    "ensure_execution_result",
]

