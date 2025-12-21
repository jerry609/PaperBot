"""
学者追踪工作流

封装论文分析流水线的高层接口，使用声明式 Pipeline 驱动。
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from typing import TYPE_CHECKING

# Pipeline 用于类型提示和未来扩展
# from paperbot.core.pipeline import Pipeline
from paperbot.domain.paper import PaperMeta, CodeMeta
from paperbot.domain.influence.result import InfluenceResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from paperbot.application.ports.event_log_port import EventLogPort


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
        """延迟初始化协调器（Phase-0: application wrapper boundary）"""
        if self._coordinator is None:
            try:
                from paperbot.application.workflows.scholar_pipeline import ScholarPipeline
                self._coordinator = ScholarPipeline(self.config)
            except ImportError as e:
                self.logger.error(f"Failed to import ScholarWorkflowCoordinator: {e}")
                raise
        return self._coordinator
    
    async def analyze_paper(
        self,
        paper: PaperMeta,
        scholar_name: Optional[str] = None,
        persist_report: bool = True,
        *,
        event_log: Optional["EventLogPort"] = None,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
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
        return await coordinator.analyze_paper(
            paper=paper,
            scholar_name=scholar_name,
            persist_report=persist_report,
            event_log=event_log,
            run_id=run_id,
            trace_id=trace_id,
        )
    
    async def analyze_papers(
        self,
        papers: List[PaperMeta],
        scholar_name: Optional[str] = None,
        *,
        event_log: Optional["EventLogPort"] = None,
        run_id: Optional[str] = None,
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
        return await coordinator.analyze_papers(
            papers=papers,
            scholar_name=scholar_name,
            event_log=event_log,
            run_id=run_id,
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
        calculator = coordinator.influence_calculator
        if calculator is None:
            return InfluenceResult(
                total_score=0.0,
                academic_score=0.0,
                engineering_score=0.0,
                explanation="InfluenceCalculator not available",
                metrics_breakdown={},
            )
        return calculator.calculate(paper, code_meta)
    
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

