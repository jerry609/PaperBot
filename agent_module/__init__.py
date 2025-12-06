"""
agent_module 已废弃，请改用 agents.*。
保留最小重定向以兼容旧引用。
"""

from agents.research_agent import ResearchAgent as PaperResearchAgent, ResearchAgent
from agents.conference_research_agent import ConferenceResearchAgent
from agents.code_analysis_agent import CodeAnalysisAgent
from agents.quality_agent import QualityAgent as QualityAssessmentAgent
from agents.documentation_agent import DocumentationAgent

__all__ = [
    "ResearchAgent",
    "PaperResearchAgent",
    "ConferenceResearchAgent",
    "CodeAnalysisAgent",
    "QualityAssessmentAgent",
    "DocumentationAgent",
]

