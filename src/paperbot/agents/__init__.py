# src/paperbot/agents/__init__.py
"""
PaperBot Agent 模块。

提供各类 AI Agent 实现：
- BaseAgent: Agent 基类
- ResearchAgent: 论文研究 Agent
- CodeAnalysisAgent: 代码分析 Agent
- QualityAgent: 质量评估 Agent
- DocumentationAgent: 文档生成 Agent
- ConferenceResearchAgent: 会议论文抓取 Agent
- ReviewerAgent: 论文评审 Agent
- VerificationAgent: 声明验证 Agent
"""

from .base import BaseAgent
from .research.agent import ResearchAgent
from .code_analysis.agent import CodeAnalysisAgent
from .quality.agent import QualityAgent
from .documentation.agent import DocumentationAgent
from .conference.agent import ConferenceResearchAgent
from .review.agent import ReviewerAgent
from .verification.agent import VerificationAgent

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "CodeAnalysisAgent",
    "QualityAgent",
    "DocumentationAgent",
    "ConferenceResearchAgent",
    "ReviewerAgent",
    "VerificationAgent",
]
