"""
影响力计算节点

计算学者的学术影响力和工程影响力
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from paperbot.repro.nodes.base_node import BaseNode


class InfluenceCalculationNode(BaseNode):
    """影响力计算节点"""
    
    def __init__(
        self,
        calculator,
        llm_client: Any = None,
        node_name: str = "InfluenceCalculationNode"
    ):
        """
        初始化节点
        
        Args:
            calculator: 影响力计算器
            llm_client: 可选的 LLM 客户端
            node_name: 节点名称
        """
        super().__init__(node_name=node_name, llm_client=llm_client)
        self.calculator = calculator
    
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        计算影响力
        
        Args:
            input_data: 包含 scholars 和 papers 的字典
            
        Returns:
            包含影响力结果的字典
        """
        scholars = input_data.get("scholars", [])
        all_papers = input_data.get("papers", [])
        
        if not scholars:
            self.log_warning("没有提供学者信息")
            return {"influence_results": []}
        
        self.log_info(f"开始计算 {len(scholars)} 位学者的影响力")
        
        results = []
        
        for scholar in scholars:
            scholar_id = scholar.get("scholar_id") or scholar.get("authorId")
            scholar_name = scholar.get("name", scholar_id)
            
            # 获取该学者的论文
            scholar_papers = self._get_scholar_papers(scholar_name, all_papers)
            
            try:
                influence = self.calculator.calculate(
                    scholar_id=scholar_id,
                    papers=scholar_papers,
                    h_index=scholar.get("h_index", scholar.get("hIndex", 0)),
                    citation_count=scholar.get("citation_count", scholar.get("citationCount", 0)),
                )
                
                result = {
                    "scholar_id": scholar_id,
                    "scholar_name": scholar_name,
                    **influence,
                }
                results.append(result)
                
                self.log_info(
                    f"  {scholar_name}: "
                    f"学术={influence.get('academic_score', 0):.2f}, "
                    f"工程={influence.get('engineering_score', 0):.2f}, "
                    f"总分={influence.get('total_score', 0):.2f}"
                )
                
            except Exception as e:
                self.log_error(f"  计算失败 {scholar_name}: {e}")
                results.append({
                    "scholar_id": scholar_id,
                    "scholar_name": scholar_name,
                    "error": str(e),
                    "academic_score": 0,
                    "engineering_score": 0,
                    "total_score": 0,
                })
        
        # 按总分排序
        results.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        
        self.log_success(f"影响力计算完成")
        
        return {
            "influence_results": results,
            "top_scholars": results[:5],
        }
    
    def _get_scholar_papers(
        self, 
        scholar_name: str, 
        all_papers: List[Dict]
    ) -> List[Dict]:
        """获取学者的论文"""
        scholar_papers = []
        
        for paper in all_papers:
            authors = paper.get("authors", [])
            # 检查作者列表
            for author in authors:
                if isinstance(author, str):
                    if scholar_name.lower() in author.lower():
                        scholar_papers.append(paper)
                        break
                elif isinstance(author, dict):
                    if scholar_name.lower() in author.get("name", "").lower():
                        scholar_papers.append(paper)
                        break
        
        return scholar_papers
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if not isinstance(input_data, dict):
            return False
        return True
