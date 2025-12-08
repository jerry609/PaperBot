# paperbot/__init__.py
"""
PaperBot - 顶会论文分析与学者追踪框架

一个专为计算机领域设计的智能论文分析框架，支持：
- 学者追踪与智能分析
- 顶会论文获取
- 代码深度分析
- 深度评审 (ReviewerAgent)
- 科学声明验证 (VerificationAgent)
- Paper2Code 代码生成 (ReproAgent)
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保 src 目录在 Python 路径中
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

__version__ = "2.0.0"
__author__ = "PaperBot Team"

# 延迟导入以避免循环依赖
def __getattr__(name: str):
    """延迟导入模块"""
    
    # Core 模块
    if name == "Pipeline":
        from paperbot.core.pipeline import Pipeline
        return Pipeline
    if name == "PipelineStage":
        from paperbot.core.pipeline import PipelineStage
        return PipelineStage
    if name == "ExecutionResult":
        from paperbot.core.abstractions import ExecutionResult
        return ExecutionResult
    if name == "Executable":
        from paperbot.core.abstractions import Executable
        return Executable
    if name == "Container":
        from paperbot.core.di import Container
        return Container
    if name == "inject":
        from paperbot.core.di import inject
        return inject
    if name == "LLMClient":
        from paperbot.core.llm_client import LLMClient
        return LLMClient
    
    # Domain 模块
    if name == "PaperMeta":
        from paperbot.domain.paper import PaperMeta
        return PaperMeta
    if name == "CodeMeta":
        from paperbot.domain.paper import CodeMeta
        return CodeMeta
    if name == "Scholar":
        from paperbot.domain.scholar import Scholar
        return Scholar
    if name == "InfluenceResult":
        from paperbot.domain.influence.result import InfluenceResult
        return InfluenceResult
    if name == "InfluenceCalculator":
        from paperbot.domain.influence.calculator import InfluenceCalculator
        return InfluenceCalculator
    
    # Workflows 模块
    if name == "ScholarTrackingWorkflow":
        from paperbot.workflows.scholar_tracking import ScholarTrackingWorkflow
        return ScholarTrackingWorkflow
    if name == "ScholarWorkflowCoordinator":
        from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator
        return ScholarWorkflowCoordinator
    
    # Agents 模块
    if name == "BaseAgent":
        from paperbot.agents.base import BaseAgent
        return BaseAgent
    if name == "ResearchAgent":
        from paperbot.agents.research.agent import ResearchAgent
        return ResearchAgent
    if name == "CodeAnalysisAgent":
        from paperbot.agents.code_analysis.agent import CodeAnalysisAgent
        return CodeAnalysisAgent
    if name == "QualityAgent":
        from paperbot.agents.quality.agent import QualityAgent
        return QualityAgent
    if name == "ReviewerAgent":
        from paperbot.agents.review.agent import ReviewerAgent
        return ReviewerAgent
    if name == "VerificationAgent":
        from paperbot.agents.verification.agent import VerificationAgent
        return VerificationAgent
    if name == "ConferenceResearchAgent":
        from paperbot.agents.conference.agent import ConferenceResearchAgent
        return ConferenceResearchAgent
    
    # Repro 模块
    if name == "ReproAgent":
        from paperbot.repro.repro_agent import ReproAgent
        return ReproAgent
    if name == "PaperContext":
        from paperbot.repro.models import PaperContext
        return PaperContext
    
    raise AttributeError(f"module 'paperbot' has no attribute '{name}'")


__all__ = [
    # 版本
    "__version__",
    
    # Core
    "Pipeline",
    "PipelineStage",
    "ExecutionResult",
    "Executable",
    "Container",
    "inject",
    "LLMClient",
    
    # Domain
    "PaperMeta",
    "CodeMeta",
    "Scholar",
    "InfluenceResult",
    "InfluenceCalculator",
    
    # Workflows
    "ScholarTrackingWorkflow",
    "ScholarWorkflowCoordinator",
    
    # Agents
    "BaseAgent",
    "ResearchAgent",
    "CodeAnalysisAgent",
    "QualityAgent",
    "ReviewerAgent",
    "VerificationAgent",
    "ConferenceResearchAgent",
    
    # Repro
    "ReproAgent",
    "PaperContext",
]
