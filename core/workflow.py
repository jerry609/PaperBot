# securipaperbot/core/workflow.py

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from pathlib import Path
import json

from agents import (
    ResearchAgent,
    CodeAnalysisAgent,
    QualityAgent,
    DocumentationAgent
)
from core.context import AnalysisContext
from utils.logger import setup_logger


class WorkflowCoordinator:
    """工作流协调器，负责协调各个代理的工作"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_logger(__name__)

        # 初始化代理
        self.research_agent = ResearchAgent(config)
        self.code_analysis_agent = CodeAnalysisAgent(config)
        self.quality_agent = QualityAgent(config)
        self.documentation_agent = DocumentationAgent(config)

        # 初始化上下文
        self.context = AnalysisContext()

        # 创建输出目录
        self.output_dir = Path(self.config.get('output_path', './output'))
        self.output_dir.mkdir(exist_ok=True)

    async def process_papers(self,
                             conference: str,
                             year: str) -> Dict[str, Any]:
        """处理指定会议和年份的论文"""
        try:
            # 1. 初始化上下文
            self.context.start_time = datetime.now()
            self.context.conference = conference
            self.context.year = year

            # 2. 论文研究阶段
            self.logger.info(f"Starting paper research for {conference} {year}")
            research_results = await self.research_agent.process(conference, year)
            self.context.update_research_results(research_results)

            # 3. 代码分析阶段
            if research_results.get('papers'):
                self.logger.info("Starting code analysis")
                for paper in research_results['papers']:
                    if paper.get('github_links'):
                        analysis_results = await self.code_analysis_agent.process(
                            paper['github_links']
                        )
                        self.context.update_analysis_results(
                            paper['title'],
                            analysis_results
                        )

            # 4. 质量评估阶段
            self.logger.info("Starting quality assessment")
            quality_results = await self.quality_agent.process(
                self.context.analysis_results
            )
            self.context.update_quality_results(quality_results)

            # 5. 文档生成阶段
            self.logger.info("Generating documentation")
            documentation = await self.documentation_agent.process(
                self.context.analysis_results,
                self.context.quality_results
            )
            self.context.update_documentation(documentation)

            # 6. 保存结果
            await self._save_results()

            return self._generate_summary()

        except Exception as e:
            self.logger.error(f"Workflow error: {str(e)}")
            raise

    async def _save_results(self):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存上下文数据
            context_path = self.output_dir / f"context_{timestamp}.json"
            with open(context_path, 'w', encoding='utf-8') as f:
                json.dump(self.context.to_dict(), f, indent=2)

            # 保存文档
            if self.context.documentation:
                doc_path = self.output_dir / f"documentation_{timestamp}.{self.config['output_format']}"
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(self.context.documentation['content'])

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def _generate_summary(self) -> Dict[str, Any]:
        """生成分析总结"""
        return {
            'conference': self.context.conference,
            'year': self.context.year,
            'papers_analyzed': len(self.context.research_results.get('papers', [])),
            'repositories_analyzed': len(self.context.analysis_results),
            'average_quality_score': self._calculate_average_quality(),
            'documentation_sections': list(self.context.documentation.get('sections', [])),
            'processing_time': (datetime.now() - self.context.start_time).total_seconds()
        }

    def _calculate_average_quality(self) -> float:
        """计算平均质量分数"""
        if not self.context.quality_results:
            return 0.0

        scores = []
        for score in self.context.quality_results.get('quality_scores', {}).values():
            if score and 'overall_score' in score:
                scores.append(score['overall_score'])

        return sum(scores) / len(scores) if scores else 0.0

    def validate_config(self) -> bool:
        """验证配置"""
        required_keys = {
            'output_path',
            'output_format',
            'github_token',
            'analysis_depth'
        }
        return all(key in self.config for key in required_keys)

    async def cancel_workflow(self):
        """取消正在进行的工作流"""
        # 实现取消逻辑
        pass

    async def get_status(self) -> Dict[str, Any]:
        """获取当前工作流状态"""
        return {
            'conference': self.context.conference,
            'year': self.context.year,
            'start_time': self.context.start_time,
            'current_stage': self.context.current_stage,
            'progress': self.context.progress,
            'errors': self.context.errors
        }