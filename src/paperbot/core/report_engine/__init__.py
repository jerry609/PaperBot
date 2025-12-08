"""
Report Engine 模块入口。

提供报告生成的核心功能，包括：
- ReportEngine: 主入口 Facade
- ReportEngineConfig: 配置
- ReportResult: 生成结果
"""

from .config import ReportEngineConfig, ReportResult
from .engine import ReportEngine
from .llm_strategy import LLMStrategy
from .compare import CompareItem, normalize_compare_items
from .view_model import ViewModel, ViewModelBuilder, SectionView

__all__ = [
    "ReportEngine",
    "ReportEngineConfig",
    "ReportResult",
    "LLMStrategy",
    "CompareItem",
    "normalize_compare_items",
    "ViewModel",
    "ViewModelBuilder",
    "SectionView",
]

