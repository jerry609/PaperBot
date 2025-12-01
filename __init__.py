# securipaperbot/__init__.py

"""
SecuriPaperBot - 智能论文分析框架

专为计算机安全领域设计的智能论文分析框架，
支持四大安全顶会论文的自动化获取和深度分析。
"""

__version__ = "0.1.0"
__author__ = "SecuriPaperBot Team"
__email__ = "contact@securipaperbot.com"

from .core import WorkflowCoordinator, AnalysisContext
from .agents import (
    BaseAgent,
    ResearchAgent,
    CodeAnalysisAgent, 
    QualityAgent,
    DocumentationAgent
)
from .utils import setup_logger, PaperDownloader
from .config import settings, Settings

# 主要导出
__all__ = [
    # 版本信息
    '__version__',
    '__author__', 
    '__email__',
    
    # 核心组件
    'WorkflowCoordinator',
    'AnalysisContext',
    
    # 代理组件
    'BaseAgent',
    'ResearchAgent',
    'CodeAnalysisAgent',
    'QualityAgent', 
    'DocumentationAgent',
    
    # 工具组件
    'setup_logger',
    'PaperDownloader',
    
    # 配置
    'settings',
    'Settings',
    
    # 主要类（兼容性）
    'PaperAnalyzer',
    'SecuriPaperBot'
]


# 为了向后兼容，提供一些别名
class PaperAnalyzer:
    """论文分析器（兼容性类）"""
    def __init__(self, config=None):
        self.coordinator = WorkflowCoordinator(config)
    
    async def analyze_paper(self, paper_path: str):
        """分析论文"""
        # 实现基础的论文分析逻辑
        return await self.coordinator.process_papers("", "")


class SecuriPaperBot:
    """SecuriPaperBot主类（兼容性类）"""
    def __init__(self, config=None):
        self.coordinator = WorkflowCoordinator(config)
        self.analyzer = PaperAnalyzer(config)
    
    async def fetch_papers(self, conference: str, year: str):
        """获取论文"""
        return await self.coordinator.process_papers(conference, year)
    
    async def analyze_paper(self, paper_path: str):
        """分析论文"""
        return await self.analyzer.analyze_paper(paper_path)
    
    def save_analysis(self, analysis):
        """保存分析结果"""
        # 实现保存逻辑
        pass


def get_version():
    """获取版本信息"""
    return __version__


def get_info():
    """获取包信息"""
    return {
        'name': 'securipaperbot',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': '智能论文分析框架，专注于安全领域四大顶会论文的自动化获取和深度分析'
    }