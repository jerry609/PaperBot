"""
学者追踪工作流

封装论文分析流水线的高层接口，使用声明式 Pipeline 驱动。
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from paperbot.core.pipeline import Pipeline, PipelineStage
from paperbot.core.abstractions import ExecutionResult
from paperbot.domain.paper import PaperMeta, CodeMeta
from paperbot.domain.influence.result import InfluenceResult

logger = logging.getLogger(__name__)


class ScholarTrackingWorkflow:
    """
    学者追踪工作流
    
    提供简化的高层接口，内部委托给 ScholarWorkflowCoordinator。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化工作流
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 延迟导入以避免循环依赖
        self._coordinator = None
    
    def _get_coordinator(self):
        """延迟初始化协调器"""
        if self._coordinator is None:
            try:
                from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator
                self._coordinator = ScholarWorkflowCoordinator(self.config)
            except ImportError as e:
                self.logger.error(f"Failed to import ScholarWorkflowCoordinator: {e}")
                raise
        return self._coordinator
    
    async def analyze_paper(
        self,
        paper: PaperMeta,
        scholar_name: Optional[str] = None,
        persist_report: bool = True,
    ) -> Tuple[Optional[Path], InfluenceResult, Dict[str, Any]]:
        """
        分析单篇论文
        
        Args:
            paper: 论文元数据
            scholar_name: 学者名称
            persist_report: 是否持久化报告
            
        Returns:
            (报告路径, 影响力结果, 流水线数据)
        """
        coordinator = self._get_coordinator()
        return await coordinator.run_paper_pipeline(
            paper=paper,
            scholar_name=scholar_name,
            persist_report=persist_report,
        )
    
    async def analyze_papers(
        self,
        papers: List[PaperMeta],
        scholar_name: Optional[str] = None,
    ) -> List[Tuple[Optional[Path], InfluenceResult, Dict[str, Any]]]:
        """
        批量分析论文
        
        Args:
            papers: 论文列表
            scholar_name: 学者名称
            
        Returns:
            结果列表
        """
        coordinator = self._get_coordinator()
        return await coordinator.run_batch_pipeline(
            papers=papers,
            scholar_name=scholar_name,
        )
    
    async def quick_score(
        self,
        paper: PaperMeta,
        code_meta: Optional[CodeMeta] = None,
    ) -> InfluenceResult:
        """
        快速计算论文影响力评分（不运行完整流水线）
        
        Args:
            paper: 论文元数据
            code_meta: 代码元数据（可选）
            
        Returns:
            影响力评分结果
        """
        coordinator = self._get_coordinator()
        return coordinator.influence_calculator.calculate(paper, code_meta)
    
    def create_paper_from_dict(self, data: Dict[str, Any]) -> PaperMeta:
        """
        从字典创建 PaperMeta
        
        Args:
            data: 论文数据字典
            
        Returns:
            PaperMeta 实例
        """
        return PaperMeta.from_dict(data)
    
    def create_paper_from_s2(self, data: Dict[str, Any]) -> PaperMeta:
        """
        从 Semantic Scholar API 响应创建 PaperMeta
        
        Args:
            data: S2 API 响应
            
        Returns:
            PaperMeta 实例
        """
        return PaperMeta.from_semantic_scholar(data)

