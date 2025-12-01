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
]