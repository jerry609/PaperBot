# core/workflow_coordinator.py
"""
MVP 版工作流协调器
用于串联多 Agent 完成学者追踪论文分析流水线
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from agents import (
    ResearchAgent,
    CodeAnalysisAgent,
    QualityAgent,
    DocumentationAgent,
)
from scholar_tracking.models import PaperMeta, CodeMeta
from scholar_tracking.models.influence import InfluenceResult
from influence import InfluenceCalculator

logger = logging.getLogger(__name__)


class ScholarWorkflowCoordinator:
    """
    学者追踪工作流协调器 (MVP 版)
    
    顺序执行流水线:
    1. ResearchAgent → 扩展摘要 + 代码仓库链接
    2. CodeAnalysisAgent → 代码质量分析
    3. QualityAgent → 综合质量评价
    4. InfluenceCalculator → PIS 评分
    5. DocumentationAgent → 生成 Markdown 报告
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化协调器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化 Agents
        self.research_agent = ResearchAgent(config)
        self.code_analysis_agent = CodeAnalysisAgent(config)
        self.quality_agent = QualityAgent(config)
        self.documentation_agent = DocumentationAgent(config)
        
        # 初始化影响力计算器
        self.influence_calculator = InfluenceCalculator(config)
    
    async def run_paper_pipeline(
        self,
        paper: PaperMeta,
        scholar_name: Optional[str] = None,
    ) -> Tuple[str, InfluenceResult, Dict[str, Any]]:
        """
        运行论文分析流水线
        
        Args:
            paper: 论文元数据
            scholar_name: 学者名称（用于报告生成）
            
        Returns:
            (report_markdown, influence_result, pipeline_data)
        """
        pipeline_data = {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "scholar_name": scholar_name,
            "started_at": datetime.now().isoformat(),
            "stages": {},
            "errors": [],
        }
        
        code_meta = None
        research_result = {}
        code_analysis_result = {}
        quality_result = {}
        
        try:
            # Stage 1: Research Agent - 扩展摘要和代码仓库发现
            self.logger.info(f"[1/5] Running ResearchAgent for: {paper.title[:50]}...")
            try:
                research_result = await self._run_research_stage(paper)
                pipeline_data["stages"]["research"] = {
                    "status": "success",
                    "result": research_result,
                }
                
                # 从研究结果中提取代码仓库信息
                if research_result.get("github_url"):
                    paper.github_url = research_result["github_url"]
                    paper.has_code = True
            except Exception as e:
                self.logger.warning(f"ResearchAgent failed: {e}")
                pipeline_data["stages"]["research"] = {"status": "failed", "error": str(e)}
                pipeline_data["errors"].append(f"Research: {e}")
            
            # Stage 2: Code Analysis Agent - 代码分析
            self.logger.info(f"[2/5] Running CodeAnalysisAgent...")
            if paper.github_url or paper.has_code:
                try:
                    code_analysis_result = await self._run_code_analysis_stage(paper)
                    pipeline_data["stages"]["code_analysis"] = {
                        "status": "success",
                        "result": code_analysis_result,
                    }
                    
                    # 构建 CodeMeta
                    code_meta = self._build_code_meta(paper, code_analysis_result)
                except Exception as e:
                    self.logger.warning(f"CodeAnalysisAgent failed: {e}")
                    pipeline_data["stages"]["code_analysis"] = {"status": "failed", "error": str(e)}
                    pipeline_data["errors"].append(f"CodeAnalysis: {e}")
            else:
                pipeline_data["stages"]["code_analysis"] = {"status": "skipped", "reason": "no code"}
            
            # Stage 3: Quality Agent - 质量评估
            self.logger.info(f"[3/5] Running QualityAgent...")
            try:
                quality_result = await self._run_quality_stage(
                    paper, research_result, code_analysis_result
                )
                pipeline_data["stages"]["quality"] = {
                    "status": "success",
                    "result": quality_result,
                }
            except Exception as e:
                self.logger.warning(f"QualityAgent failed: {e}")
                pipeline_data["stages"]["quality"] = {"status": "failed", "error": str(e)}
                pipeline_data["errors"].append(f"Quality: {e}")
            
            # Stage 4: Influence Calculator - 影响力评分
            self.logger.info(f"[4/5] Calculating influence score...")
            influence_result = self.influence_calculator.calculate(paper, code_meta)
            pipeline_data["stages"]["influence"] = {
                "status": "success",
                "result": influence_result.to_dict(),
            }
            
            # Stage 5: Documentation Agent - 报告生成
            self.logger.info(f"[5/5] Generating report...")
            try:
                report_markdown = await self._generate_report(
                    paper=paper,
                    scholar_name=scholar_name,
                    research_result=research_result,
                    code_analysis_result=code_analysis_result,
                    quality_result=quality_result,
                    influence_result=influence_result,
                )
                pipeline_data["stages"]["documentation"] = {"status": "success"}
            except Exception as e:
                self.logger.warning(f"DocumentationAgent failed: {e}")
                pipeline_data["stages"]["documentation"] = {"status": "failed", "error": str(e)}
                pipeline_data["errors"].append(f"Documentation: {e}")
                # 使用备用报告生成
                report_markdown = self._generate_fallback_report(
                    paper, influence_result, pipeline_data
                )
            
            pipeline_data["completed_at"] = datetime.now().isoformat()
            pipeline_data["status"] = "success" if not pipeline_data["errors"] else "partial"
            
            return report_markdown, influence_result, pipeline_data
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {paper.title}: {e}")
            pipeline_data["status"] = "failed"
            pipeline_data["errors"].append(str(e))
            
            # 即使失败也计算影响力分数
            influence_result = self.influence_calculator.calculate(paper, None)
            report_markdown = self._generate_fallback_report(paper, influence_result, pipeline_data)
            
            return report_markdown, influence_result, pipeline_data
    
    async def _run_research_stage(self, paper: PaperMeta) -> Dict[str, Any]:
        """运行研究阶段"""
        # 调用 ResearchAgent 获取更多论文信息
        result = await self.research_agent.process(
            paper_title=paper.title,
            paper_id=paper.paper_id,
            abstract=paper.abstract,
        )
        return result if isinstance(result, dict) else {"raw": str(result)}
    
    async def _run_code_analysis_stage(self, paper: PaperMeta) -> Dict[str, Any]:
        """运行代码分析阶段"""
        if not paper.github_url:
            return {"status": "no_code_url"}
        
        result = await self.code_analysis_agent.process(
            repo_url=paper.github_url,
        )
        return result if isinstance(result, dict) else {"raw": str(result)}
    
    async def _run_quality_stage(
        self,
        paper: PaperMeta,
        research_result: Dict[str, Any],
        code_analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """运行质量评估阶段"""
        # 合并上下文信息
        context = {
            "paper": paper.to_dict(),
            "research": research_result,
            "code_analysis": code_analysis_result,
        }
        
        result = await self.quality_agent.process(context)
        return result if isinstance(result, dict) else {"raw": str(result)}
    
    def _build_code_meta(
        self,
        paper: PaperMeta,
        code_analysis_result: Dict[str, Any],
    ) -> Optional[CodeMeta]:
        """从代码分析结果构建 CodeMeta"""
        if not paper.github_url:
            return None
        
        try:
            return CodeMeta(
                repo_url=paper.github_url,
                repo_name=code_analysis_result.get("repo_name"),
                stars=code_analysis_result.get("stars", 0),
                forks=code_analysis_result.get("forks", 0),
                language=code_analysis_result.get("language"),
                updated_at=code_analysis_result.get("updated_at"),
                has_readme=code_analysis_result.get("has_readme", False),
                reproducibility_score=code_analysis_result.get("reproducibility_score"),
            )
        except Exception as e:
            self.logger.warning(f"Failed to build CodeMeta: {e}")
            return CodeMeta(repo_url=paper.github_url)
    
    async def _generate_report(
        self,
        paper: PaperMeta,
        scholar_name: Optional[str],
        research_result: Dict[str, Any],
        code_analysis_result: Dict[str, Any],
        quality_result: Dict[str, Any],
        influence_result: InfluenceResult,
    ) -> str:
        """生成完整的 Markdown 报告"""
        # 调用 DocumentationAgent 生成报告
        report_data = {
            "paper": paper.to_dict(),
            "scholar_name": scholar_name,
            "research": research_result,
            "code_analysis": code_analysis_result,
            "quality": quality_result,
            "influence": influence_result.to_dict(),
        }
        
        result = await self.documentation_agent.process(report_data)
        
        if isinstance(result, dict) and "report" in result:
            return result["report"]
        elif isinstance(result, str):
            return result
        else:
            # 使用备用模板
            return self._generate_fallback_report(paper, influence_result, {})
    
    def _generate_fallback_report(
        self,
        paper: PaperMeta,
        influence_result: InfluenceResult,
        pipeline_data: Dict[str, Any],
    ) -> str:
        """生成备用报告（当 DocumentationAgent 失败时）"""
        authors = ", ".join(paper.authors) if paper.authors else "未知"
        
        report = f"""# {paper.title}

## 📋 元信息

| 属性 | 值 |
|------|-----|
| 作者 | {authors} |
| 年份 | {paper.year or '未知'} |
| 发表于 | {paper.venue or '未知'} |
| 引用数 | {paper.citation_count} |
| Semantic Scholar ID | {paper.paper_id} |

## 📝 摘要

{paper.abstract or paper.tldr or '暂无摘要'}

## 💻 代码信息

"""
        if paper.github_url:
            report += f"- **仓库地址**: [{paper.github_url}]({paper.github_url})\n"
        else:
            report += "未发现公开代码仓库\n"
        
        report += f"""
## 📊 影响力评分 (PIS)

| 指标 | 分数 |
|------|------|
| **总分** | {influence_result.total_score:.1f}/100 |
| 学术影响力 | {influence_result.academic_score:.1f}/100 |
| 工程影响力 | {influence_result.engineering_score:.1f}/100 |

{influence_result.explanation}

## 🎯 推荐级别

**{influence_result.recommendation.value}**

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*由 PaperBot 自动生成*
"""
        return report
    
    async def run_batch_pipeline(
        self,
        papers: List[PaperMeta],
        scholar_name: Optional[str] = None,
    ) -> List[Tuple[str, InfluenceResult, Dict[str, Any]]]:
        """
        批量运行论文分析流水线
        
        Args:
            papers: 论文列表
            scholar_name: 学者名称
            
        Returns:
            结果列表
        """
        results = []
        total = len(papers)
        
        for i, paper in enumerate(papers, 1):
            self.logger.info(f"Processing paper {i}/{total}: {paper.title[:50]}...")
            
            try:
                result = await self.run_paper_pipeline(paper, scholar_name)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process paper: {e}")
                # 创建失败结果
                influence = self.influence_calculator.calculate(paper, None)
                report = self._generate_fallback_report(paper, influence, {"error": str(e)})
                results.append((report, influence, {"status": "failed", "error": str(e)}))
        
        return results
