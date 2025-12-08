"""
Agent 层 - 所有智能代理的入口。
"""

from .base import BaseAgent
from .mixins import SemanticScholarMixin, TextParsingMixin, JSONParserMixin, JSONParseError

# 为了兼容，也从根目录 agents 导出具体实现
try:
    from agents import (
        ResearchAgent,
        CodeAnalysisAgent,
        QualityAgent,
        DocumentationAgent,
        ReviewerAgent,
        VerificationAgent,
        ConferenceResearchAgent,
    )
except ImportError:
    ResearchAgent = None
    CodeAnalysisAgent = None
    QualityAgent = None
    DocumentationAgent = None
    ReviewerAgent = None
    VerificationAgent = None
    ConferenceResearchAgent = None

__all__ = [
    # 基类
    "BaseAgent",
    # Mixins
    "SemanticScholarMixin",
    "TextParsingMixin",
    "JSONParserMixin",
    "JSONParseError",
    # 具体 Agent（兼容）
    "ResearchAgent",
    "CodeAnalysisAgent",
    "QualityAgent",
    "DocumentationAgent",
    "ReviewerAgent",
    "VerificationAgent",
    "ConferenceResearchAgent",
]
