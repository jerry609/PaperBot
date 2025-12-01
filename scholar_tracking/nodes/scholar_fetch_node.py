"""
学者信息获取节点

从 Semantic Scholar API 获取学者信息
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from core.base_node import BaseNode


class ScholarFetchNode(BaseNode):
    """学者信息获取节点"""
    
    def __init__(
        self,
        scholar_profile,
        llm_client: Any = None,
        node_name: str = "ScholarFetchNode"
    ):
        """
        初始化节点
        
        Args:
            scholar_profile: 学者档案代理
            llm_client: 可选的 LLM 客户端
            node_name: 节点名称
        """
        super().__init__(node_name=node_name, llm_client=llm_client)
        self.scholar_profile = scholar_profile
    
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        获取学者信息
        
        Args:
            input_data: 包含 scholar_ids 的字典
            
        Returns:
            包含学者信息列表的字典
        """
        scholar_ids = input_data.get("scholar_ids", [])
        
        if not scholar_ids:
            self.log_warning("没有提供学者ID")
            return {"scholars": []}
        
        self.log_info(f"开始获取 {len(scholar_ids)} 位学者的信息")
        
        scholars = []
        for scholar_id in scholar_ids:
            try:
                scholar_data = self.scholar_profile.get_scholar(scholar_id)
                if scholar_data:
                    scholars.append(scholar_data)
                    self.log_info(f"  ✓ 获取学者: {scholar_data.get('name', scholar_id)}")
                else:
                    self.log_warning(f"  ✗ 无法获取学者: {scholar_id}")
            except Exception as e:
                self.log_error(f"  ✗ 获取学者失败 {scholar_id}: {e}")
        
        self.log_success(f"成功获取 {len(scholars)}/{len(scholar_ids)} 位学者信息")
        
        return {"scholars": scholars}
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if not isinstance(input_data, dict):
            return False
        return True
