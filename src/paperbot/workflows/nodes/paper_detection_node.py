"""
论文检测节点

检测学者的新论文
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

from paperbot.repro.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class PaperDetectionNode(BaseNode):
    """论文检测节点"""
    
    def __init__(
        self,
        paper_tracker,
        node_name: str = "PaperDetectionNode"
    ):
        """
        初始化节点
        
        Args:
            paper_tracker: 论文追踪代理
            node_name: 节点名称
        """
        super().__init__(node_name=node_name)
        self.paper_tracker = paper_tracker
    
    async def _execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        检测新论文
        
        Args:
            input_data: 包含 scholars 和 since_days 的字典
            
        Returns:
            包含新论文列表的字典
        """
        scholars = input_data.get("scholars", [])
        since_days = input_data.get("since_days", 30)
        
        if not scholars:
            logger.warning("没有提供学者信息")
            return {"new_papers": [], "papers_by_scholar": {}}
        
        logger.info(f"开始检测 {len(scholars)} 位学者的新论文 (最近 {since_days} 天)")
        
        all_papers = []
        papers_by_scholar = {}
        
        for scholar in scholars:
            scholar_id = scholar.get("scholar_id") or scholar.get("authorId")
            scholar_name = scholar.get("name", scholar_id)
            
            if not scholar_id:
                logger.warning(f"学者缺少ID: {scholar}")
                continue
            
            try:
                papers = self.paper_tracker.get_new_papers(
                    scholar_id, 
                    since_days=since_days
                )
                
                papers_by_scholar[scholar_name] = papers
                all_papers.extend(papers)
                
                if papers:
                    logger.info(f"  {scholar_name}: 发现 {len(papers)} 篇新论文")
                    for paper in papers[:3]:  # 只显示前3篇
                        title = paper.get("title", "Unknown")[:50]
                        logger.info(f"    - {title}...")
                    if len(papers) > 3:
                        logger.info(f"    ... 还有 {len(papers) - 3} 篇")
                else:
                    logger.info(f"  {scholar_name}: 没有新论文")
                    
            except Exception as e:
                logger.error(f"  检测失败 {scholar_name}: {e}")
                papers_by_scholar[scholar_name] = []
        
        # 去重（同一论文可能有多个作者）
        unique_papers = self._deduplicate_papers(all_papers)
        
        logger.info(f"共发现 {len(unique_papers)} 篇新论文 (去重后)")
        
        return {
            "new_papers": unique_papers,
            "papers_by_scholar": papers_by_scholar,
            "total_count": len(unique_papers),
        }
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """论文去重"""
        seen = set()
        unique = []
        
        for paper in papers:
            paper_id = paper.get("paperId") or paper.get("paper_id")
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                unique.append(paper)
        
        return unique
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if not isinstance(input_data, dict):
            return False
        return True
