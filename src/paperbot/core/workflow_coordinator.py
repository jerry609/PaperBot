# paperbot/core/workflow_coordinator.py
"""
学者追踪工作流协调器

负责协调论文分析流水线的各个阶段。
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """流水线上下文"""
    paper: Any = None
    scholar_name: Optional[str] = None
    research_result: Dict[str, Any] = field(default_factory=dict)
    code_analysis_result: Dict[str, Any] = field(default_factory=dict)
    quality_result: Dict[str, Any] = field(default_factory=dict)
    influence_result: Any = None
    stages: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error: Optional[str] = None


class ScholarWorkflowCoordinator:
    """
    学者工作流协调器
    
    协调论文分析流水线的各个阶段：
    1. 研究分析 (ResearchAgent)
    2. 代码分析 (CodeAnalysisAgent) 
    3. 质量评估 (QualityAgent)
    4. 影响力计算 (InfluenceCalculator)
    5. 报告生成 (ReportWriter)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化协调器
        
        Args:
            config: 配置字典，包含：
                - output_dir: 输出目录
                - report_template: 报告模板名称
                - use_documentation_agent: 是否使用文档Agent
                - mode: 运行模式 (production/academic)
        """
        self.config = config or {}
        self.output_dir = Path(self.config.get("output_dir", "./output/reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_template = self.config.get("report_template", "paper_report.md.j2")
        self.use_documentation_agent = self.config.get("use_documentation_agent", False)
        self.mode = self.config.get("mode", "production")
        
        # 延迟初始化组件
        self._research_agent = None
        self._code_analysis_agent = None
        self._quality_agent = None
        self._influence_calculator = None
        self._report_writer = None
        
        logger.info(f"ScholarWorkflowCoordinator initialized with mode={self.mode}")
    
    @property
    def research_agent(self):
        """延迟初始化 ResearchAgent"""
        if self._research_agent is None:
            try:
                from paperbot.agents.research.agent import ResearchAgent
                self._research_agent = ResearchAgent({})
            except ImportError as e:
                logger.warning(f"Failed to import ResearchAgent: {e}")
        return self._research_agent
    
    @property
    def code_analysis_agent(self):
        """延迟初始化 CodeAnalysisAgent"""
        if self._code_analysis_agent is None:
            try:
                from paperbot.agents.code_analysis.agent import CodeAnalysisAgent
                self._code_analysis_agent = CodeAnalysisAgent({})
            except ImportError as e:
                logger.warning(f"Failed to import CodeAnalysisAgent: {e}")
        return self._code_analysis_agent
    
    @property
    def quality_agent(self):
        """延迟初始化 QualityAgent"""
        if self._quality_agent is None:
            try:
                from paperbot.agents.quality.agent import QualityAgent
                self._quality_agent = QualityAgent({})
            except ImportError as e:
                logger.warning(f"Failed to import QualityAgent: {e}")
        return self._quality_agent
    
    @property
    def influence_calculator(self):
        """延迟初始化 InfluenceCalculator"""
        if self._influence_calculator is None:
            try:
                from paperbot.domain.influence.calculator import InfluenceCalculator
                self._influence_calculator = InfluenceCalculator()
            except ImportError as e:
                logger.warning(f"Failed to import InfluenceCalculator: {e}")
        return self._influence_calculator
    
    @property
    def report_writer(self):
        """延迟初始化 ReportWriter"""
        if self._report_writer is None:
            try:
                from paperbot.presentation.reports.writer import ReportWriter
                self._report_writer = ReportWriter(
                    template_name=self.report_template,
                    output_dir=str(self.output_dir),
                )
            except ImportError as e:
                logger.warning(f"Failed to import ReportWriter: {e}")
        return self._report_writer
    
    async def run_paper_pipeline(
        self,
        paper: Any,
        scholar_name: Optional[str] = None,
        persist_report: bool = True,
    ) -> Tuple[Optional[Path], Any, Dict[str, Any]]:
        """
        运行论文分析流水线
        
        Args:
            paper: 论文元数据 (PaperMeta)
            scholar_name: 学者名称
            persist_report: 是否持久化报告
            
        Returns:
            (报告路径, 影响力结果, 流水线数据)
        """
        ctx = PipelineContext(paper=paper, scholar_name=scholar_name)
        
        try:
            # 1. 研究分析
            ctx = await self._run_research_stage(ctx)
            
            # 2. 代码分析（如果有代码）
            if paper.github_url or getattr(paper, 'has_code', False):
                ctx = await self._run_code_analysis_stage(ctx)
            
            # 3. 质量评估
            ctx = await self._run_quality_stage(ctx)
            
            # 4. 影响力计算
            ctx = await self._run_influence_stage(ctx)
            
            # 5. 报告生成
            report_path = None
            if persist_report and self.report_writer:
                report_path = await self._run_report_stage(ctx)
            
            ctx.status = "success"
            
            return report_path, ctx.influence_result, self._build_pipeline_data(ctx)
            
        except Exception as e:
            logger.error(f"Pipeline failed for paper {paper.paper_id}: {e}")
            ctx.status = "error"
            ctx.error = str(e)
            
            # 返回默认影响力结果
            default_influence = self._create_default_influence()
            return None, default_influence, self._build_pipeline_data(ctx)
    
    async def _run_research_stage(self, ctx: PipelineContext) -> PipelineContext:
        """运行研究分析阶段"""
        if not self.research_agent:
            logger.warning("ResearchAgent not available, skipping research stage")
            return ctx
        
        try:
            result = await self.research_agent.process(
                title=ctx.paper.title,
                abstract=getattr(ctx.paper, 'abstract', '') or '',
            )
            ctx.research_result = result
            ctx.stages["research"] = {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Research stage failed: {e}")
            ctx.stages["research"] = {"status": "error", "error": str(e)}
        
        return ctx
    
    async def _run_code_analysis_stage(self, ctx: PipelineContext) -> PipelineContext:
        """运行代码分析阶段"""
        if not self.code_analysis_agent:
            logger.warning("CodeAnalysisAgent not available, skipping code analysis stage")
            return ctx
        
        github_url = getattr(ctx.paper, 'github_url', None)
        if not github_url:
            return ctx
        
        try:
            result = await self.code_analysis_agent.process(repo_url=github_url)
            ctx.code_analysis_result = result
            ctx.stages["code_analysis"] = {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Code analysis stage failed: {e}")
            ctx.stages["code_analysis"] = {"status": "error", "error": str(e)}
        
        return ctx
    
    async def _run_quality_stage(self, ctx: PipelineContext) -> PipelineContext:
        """运行质量评估阶段"""
        if not self.quality_agent:
            logger.warning("QualityAgent not available, skipping quality stage")
            return ctx
        
        try:
            result = await self.quality_agent.process(
                title=ctx.paper.title,
                abstract=getattr(ctx.paper, 'abstract', '') or '',
                research_result=ctx.research_result,
                code_analysis_result=ctx.code_analysis_result,
            )
            ctx.quality_result = result
            ctx.stages["quality"] = {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Quality stage failed: {e}")
            ctx.stages["quality"] = {"status": "error", "error": str(e)}
        
        return ctx
    
    async def _run_influence_stage(self, ctx: PipelineContext) -> PipelineContext:
        """运行影响力计算阶段"""
        if not self.influence_calculator:
            logger.warning("InfluenceCalculator not available, using default influence")
            ctx.influence_result = self._create_default_influence()
            return ctx
        
        try:
            # 构建 CodeMeta（如果有代码分析结果）
            code_meta = None
            if ctx.code_analysis_result:
                try:
                    from paperbot.domain.paper import CodeMeta
                    code_meta = CodeMeta(
                        github_url=getattr(ctx.paper, 'github_url', ''),
                        stars=ctx.code_analysis_result.get('stars', 0),
                        forks=ctx.code_analysis_result.get('forks', 0),
                        has_readme=ctx.code_analysis_result.get('has_readme', False),
                        has_tests=ctx.code_analysis_result.get('has_tests', False),
                        last_commit_date=ctx.code_analysis_result.get('last_commit_date'),
                    )
                except ImportError:
                    pass
            
            influence = self.influence_calculator.calculate(ctx.paper, code_meta)
            ctx.influence_result = influence
            ctx.stages["influence"] = {"status": "success", "result": influence}
        except Exception as e:
            logger.error(f"Influence stage failed: {e}")
            ctx.influence_result = self._create_default_influence()
            ctx.stages["influence"] = {"status": "error", "error": str(e)}
        
        return ctx
    
    async def _run_report_stage(self, ctx: PipelineContext) -> Optional[Path]:
        """运行报告生成阶段"""
        if not self.report_writer:
            return None
        
        try:
            md = self.report_writer.render_template(
                paper=ctx.paper,
                influence=ctx.influence_result,
                research_result=ctx.research_result,
                code_analysis_result=ctx.code_analysis_result,
                quality_result=ctx.quality_result,
                scholar_name=ctx.scholar_name,
            )
            
            path = self.report_writer.write_report(md, ctx.paper, scholar_name=ctx.scholar_name)
            ctx.stages["report"] = {"status": "success", "path": str(path)}
            return path
        except Exception as e:
            logger.error(f"Report stage failed: {e}")
            ctx.stages["report"] = {"status": "error", "error": str(e)}
            return None
    
    def _create_default_influence(self):
        """创建默认影响力结果"""
        try:
            from paperbot.domain.influence.result import InfluenceResult, Recommendation
            return InfluenceResult(
                total_score=0.0,
                academic_score=0.0,
                engineering_score=0.0,
                explanation="No influence data available.",
                metrics_breakdown={},
                recommendation=Recommendation.LOW,
            )
        except ImportError:
            # 返回简单字典
            return {
                "total_score": 0.0,
                "academic_score": 0.0,
                "engineering_score": 0.0,
                "explanation": "No influence data available.",
                "metrics_breakdown": {},
                "recommendation": "low",
            }
    
    def _build_pipeline_data(self, ctx: PipelineContext) -> Dict[str, Any]:
        """构建流水线数据"""
        return {
            "status": ctx.status,
            "error": ctx.error,
            "stages": ctx.stages,
            "scholar_name": ctx.scholar_name,
        }
    
    async def run_batch_pipeline(
        self,
        papers: List[Any],
        scholar_name: Optional[str] = None,
    ) -> List[Tuple[Optional[Path], Any, Dict[str, Any]]]:
        """
        批量运行论文分析流水线
        
        Args:
            papers: 论文列表
            scholar_name: 学者名称
            
        Returns:
            结果列表
        """
        results = []
        for paper in papers:
            result = await self.run_paper_pipeline(
                paper=paper,
                scholar_name=scholar_name,
                persist_report=True,
            )
            results.append(result)
        return results

