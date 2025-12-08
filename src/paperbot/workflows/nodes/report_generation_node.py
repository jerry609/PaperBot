"""
报告生成节点

生成学者追踪报告
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from paperbot.repro.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ReportGenerationNode(BaseNode):
    """报告生成节点"""
    
    def __init__(
        self,
        report_writer,
        llm_client: Any = None,
        node_name: str = "ReportGenerationNode"
    ):
        """
        初始化节点
        
        Args:
            report_writer: 报告写入器
            llm_client: 可选的 LLM 客户端（用于智能摘要生成）
            node_name: 节点名称
        """
        super().__init__(node_name=node_name)
        self.report_writer = report_writer
        self.llm_client = llm_client
    
    async def _execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        生成报告
        
        Args:
            input_data: 包含所有追踪数据的字典
            
        Returns:
            包含报告路径的字典
        """
        logger.info("开始生成报告...")
        
        # 准备报告数据
        report_data = self._prepare_report_data(input_data)
        
        # 如果有 LLM，生成智能摘要
        if self.llm_client:
            try:
                report_data["ai_summary"] = self._generate_ai_summary(report_data)
                logger.info("  AI 摘要生成完成")
            except Exception as e:
                logger.warning(f"  AI 摘要生成失败: {e}")
                report_data["ai_summary"] = None
        
        # 写入报告
        try:
            report_path = self.report_writer.write(report_data)
            logger.info(f"报告已生成: {report_path}")
            
            return {
                "report_path": report_path,
                "report_data": report_data,
                "success": True,
            }
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return {
                "report_path": None,
                "error": str(e),
                "success": False,
            }
    
    def _prepare_report_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备报告数据"""
        scholars = input_data.get("scholars", [])
        new_papers = input_data.get("new_papers", [])
        influence_results = input_data.get("influence_results", [])
        
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_scholars": len(scholars),
                "total_new_papers": len(new_papers),
                "tracking_period": input_data.get("since_days", 30),
            },
            "scholars": scholars,
            "new_papers": new_papers,
            "influence_results": influence_results,
            "highlights": self._extract_highlights(
                scholars, new_papers, influence_results
            ),
        }
    
    def _extract_highlights(
        self,
        scholars: List[Dict],
        new_papers: List[Dict],
        influence_results: List[Dict]
    ) -> Dict[str, Any]:
        """提取亮点信息"""
        highlights = {
            "top_papers": [],
            "most_active_scholars": [],
            "trending_topics": [],
        }
        
        # 按引用数排序论文
        sorted_papers = sorted(
            new_papers,
            key=lambda x: x.get("citationCount", x.get("citation_count", 0)),
            reverse=True
        )
        highlights["top_papers"] = sorted_papers[:5]
        
        # 最活跃的学者（论文最多）
        scholar_paper_count = {}
        for paper in new_papers:
            for author in paper.get("authors", []):
                name = author if isinstance(author, str) else author.get("name", "")
                if name:
                    scholar_paper_count[name] = scholar_paper_count.get(name, 0) + 1
        
        most_active = sorted(
            scholar_paper_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        highlights["most_active_scholars"] = [
            {"name": name, "paper_count": count}
            for name, count in most_active
        ]
        
        # 提取主题（从论文标题和摘要）
        # TODO: 使用 LLM 进行主题提取
        
        return highlights
    
    def _generate_ai_summary(self, report_data: Dict[str, Any]) -> str:
        """使用 LLM 生成智能摘要"""
        if not self.llm_client:
            return ""
        
        # 准备摘要 prompt
        summary_data = report_data.get("summary", {})
        highlights = report_data.get("highlights", {})
        
        system_prompt = """你是一位学术追踪助手。请根据提供的学者追踪数据生成一份简洁的中文摘要。
摘要应包括：
1. 追踪概览（学者数量、新论文数量）
2. 重要发现（值得关注的新论文）
3. 学者动态（最活跃的学者）
4. 趋势观察（如果有明显趋势）

请用3-5段话总结，语言专业但易读。"""
        
        user_prompt = f"""追踪数据：
- 追踪学者数: {summary_data.get('total_scholars', 0)}
- 新论文数: {summary_data.get('total_new_papers', 0)}
- 追踪周期: 最近 {summary_data.get('tracking_period', 30)} 天

Top 论文:
{self._format_papers_for_prompt(highlights.get('top_papers', []))}

最活跃学者:
{self._format_scholars_for_prompt(highlights.get('most_active_scholars', []))}

请生成追踪报告摘要。"""
        
        try:
            response = self.llm_client.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return ""
    
    def _format_papers_for_prompt(self, papers: List[Dict]) -> str:
        """格式化论文列表用于 prompt"""
        lines = []
        for i, paper in enumerate(papers[:5], 1):
            title = paper.get("title", "Unknown")
            citations = paper.get("citationCount", paper.get("citation_count", 0))
            venue = paper.get("venue", "Unknown")
            lines.append(f"{i}. {title} (引用: {citations}, 发表于: {venue})")
        return "\n".join(lines) if lines else "无"
    
    def _format_scholars_for_prompt(self, scholars: List[Dict]) -> str:
        """格式化学者列表用于 prompt"""
        lines = []
        for scholar in scholars[:5]:
            name = scholar.get("name", "Unknown")
            count = scholar.get("paper_count", 0)
            lines.append(f"- {name}: {count} 篇论文")
        return "\n".join(lines) if lines else "无"
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if not isinstance(input_data, dict):
            return False
        return True
