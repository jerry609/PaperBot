"""
PaperBot - 学者追踪与论文分析系统

这是新的分层架构入口点，提供向后兼容的导出。

目录结构:
- core/: 核心抽象 (Executable, Pipeline, DI, Errors)
- agents/: Agent 实现
- infrastructure/: 基础设施 (LLM, API, Storage)
- domain/: 领域模型 (Paper, Scholar, Influence)
- workflows/: 工作流定义
- presentation/: 展示层 (CLI, Reports)
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便兼容旧的导入方式
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__version__ = "1.0.0"

# 核心抽象
from .core import (
    Executable,
    ExecutionResult,
    ensure_execution_result,
    Pipeline,
    PipelineStage,
    PipelineResult,
    StageResult,
    Container,
    inject,
    bootstrap_dependencies,
    ErrorSeverity,
    PaperBotError,
    LLMError,
    APIError,
    ValidationError,
    Result,
)

# 领域模型
from .domain import (
    PaperMeta,
    CodeMeta,
    Scholar,
    InfluenceResult,
    InfluenceLevel,
)

# 工作流
from .workflows import ScholarTrackingWorkflow

# 基础设施
from .infrastructure import (
    LLMClient,
    APIClient,
    SemanticScholarClient,
    CacheService,
)

# 展示层
from .presentation import run_cli, ReportGenerator

__all__ = [
    # 版本
    "__version__",
    # 核心抽象
    "Executable",
    "ExecutionResult",
    "ensure_execution_result",
    "Pipeline",
    "PipelineStage",
    "PipelineResult",
    "StageResult",
    "Container",
    "inject",
    "bootstrap_dependencies",
    "ErrorSeverity",
    "PaperBotError",
    "LLMError",
    "APIError",
    "ValidationError",
    "Result",
    # 领域模型
    "PaperMeta",
    "CodeMeta",
    "Scholar",
    "InfluenceResult",
    "InfluenceLevel",
    # 工作流
    "ScholarTrackingWorkflow",
    # 基础设施
    "LLMClient",
    "APIClient",
    "SemanticScholarClient",
    "CacheService",
    # 展示层
    "run_cli",
    "ReportGenerator",
]
