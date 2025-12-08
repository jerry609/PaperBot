# securipaperbot/core/__init__.py

from .workflow import WorkflowCoordinator
from .context import AnalysisContext
from .llm_client import LLMClient
from .base_node import BaseNode, StateMutationNode, LLMNode
from .state import (
    TrackingState,
    TrackingStage,
    ScholarState,
    PaperState,
    InfluenceState,
)

# 新增统一抽象
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
    # 原有模块
    'WorkflowCoordinator',
    'AnalysisContext',
    # 新增 - LLM 客户端
    'LLMClient',
    # 新增 - 节点基类
    'BaseNode',
    'StateMutationNode',
    'LLMNode',
    # 新增 - 状态管理
    'TrackingState',
    'TrackingStage',
    'ScholarState',
    'PaperState',
    'InfluenceState',
    # 新增 - 统一抽象
    'Executable',
    'ExecutionResult',
    'ensure_execution_result',
    # 新增 - 流水线
    'Pipeline',
    'PipelineStage',
    'PipelineResult',
    'StageResult',
    # 新增 - 依赖注入
    'Container',
    'inject',
    'bootstrap_dependencies',
    # 新增 - 错误处理
    'ErrorSeverity',
    'PaperBotError',
    'LLMError',
    'APIError',
    'ValidationError',
    'Result',
]