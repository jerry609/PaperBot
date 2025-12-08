"""
学者信息获取节点

从 Semantic Scholar API 获取学者信息
"""

from typing import Any, Dict
import logging

from paperbot.repro.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ScholarFetchNode(BaseNode):
    """学者信息获取节点"""
    
    def __init__(
        self,
        scholar_profile,
        node_name: str = "ScholarFetchNode"
    ):
        """
        初始化节点
        
        Args:
            scholar_profile: 学者档案代理
            node_name: 节点名称
        """
        super().__init__(node_name=node_name)
        self.scholar_profile = scholar_profile
    
    async def _execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        获取学者信息
        
        Args:
            input_data: 包含 scholar_ids 的字典
            
        Returns:
            包含学者信息列表的字典
        """
        scholar_ids = input_data.get("scholar_ids", [])
        
        if not scholar_ids:
            logger.warning("没有提供学者ID")
            return {"scholars": []}
        
        logger.info(f"开始获取 {len(scholar_ids)} 位学者的信息")
        
        scholars = []
        for scholar_id in scholar_ids:
            try:
                scholar_data = self.scholar_profile.get_scholar(scholar_id)
                if scholar_data:
                    scholars.append(scholar_data)
                    logger.info(f"  ✓ 获取学者: {scholar_data.get('name', scholar_id)}")
                else:
                    logger.warning(f"  ✗ 无法获取学者: {scholar_id}")
            except Exception as e:
                logger.error(f"  ✗ 获取学者失败 {scholar_id}: {e}")
        
        logger.info(f"成功获取 {len(scholars)}/{len(scholar_ids)} 位学者信息")
        
        return {"scholars": scholars}
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if not isinstance(input_data, dict):
            return False
        return True
