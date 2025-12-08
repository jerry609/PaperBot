# scholar_tracking/agents/scholar_profile_agent.py
"""
学者画像管理 Agent
负责管理学者信息和缓存
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path

from paperbot.agents.base import BaseAgent
from paperbot.domain.scholar import Scholar
from paperbot.domain.paper import PaperMeta
from paperbot.infrastructure.services import SubscriptionService, CacheService

logger = logging.getLogger(__name__)


class ScholarProfileAgent(BaseAgent):
    """
    学者画像管理 Agent
    
    主要功能：
    - 读取和管理订阅学者配置
    - 管理学者级缓存
    - 维护学者画像信息
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 初始化服务
        config_path = None
        if config:
            config_path = (
                config.get("subscriptions_config_path")
                or config.get("subscription_config_path")
                or config.get("config_path")
            )

        self.subscription_service = SubscriptionService(config_path=config_path)
        
        # 获取缓存目录
        cache_dir = self.subscription_service.get_cache_dir()
        self.cache_service = CacheService(cache_dir)
        
        # 学者列表缓存
        self._scholars: Optional[List[Scholar]] = None
    
    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """执行学者画像管理操作"""
        action = kwargs.get("action", "list")
        
        if action == "list":
            scholars = self.list_tracked_scholars()
            return {"scholars": [s.to_dict() for s in scholars]}
        elif action == "get_cache":
            scholar_id = kwargs.get("scholar_id")
            if scholar_id:
                paper_ids = self.load_scholar_cache(scholar_id)
                return {"paper_ids": list(paper_ids)}
        
        return {"error": "Unknown action"}
    
    def list_tracked_scholars(self) -> List[Scholar]:
        """
        获取所有订阅的学者列表
        
        Returns:
            学者对象列表
        """
        if self._scholars is None:
            self._scholars = self.subscription_service.get_scholars()
        return self._scholars
    
    def get_scholar_by_id(self, scholar_id: str) -> Optional[Scholar]:
        """
        根据 ID 获取学者
        
        Args:
            scholar_id: Semantic Scholar ID
            
        Returns:
            学者对象，如果不存在则返回 None
        """
        scholars = self.list_tracked_scholars()
        for scholar in scholars:
            if scholar.semantic_scholar_id == scholar_id:
                return scholar
        return None
    
    def load_scholar_cache(self, scholar_id: str) -> Set[str]:
        """
        加载学者的已处理论文 ID 集合
        
        Args:
            scholar_id: Semantic Scholar ID
            
        Returns:
            已处理的论文 ID 集合
        """
        return self.cache_service.load_paper_ids(scholar_id)
    
    def save_scholar_cache(self, scholar_id: str, paper_ids: Set[str]):
        """
        保存学者的已处理论文 ID 集合
        
        Args:
            scholar_id: Semantic Scholar ID
            paper_ids: 论文 ID 集合
        """
        self.cache_service.save_paper_ids(scholar_id, paper_ids)
    
    def add_to_cache(self, scholar_id: str, new_paper_ids: Set[str]) -> Set[str]:
        """
        添加新论文 ID 到缓存
        
        Args:
            scholar_id: Semantic Scholar ID
            new_paper_ids: 新论文 ID 集合
            
        Returns:
            更新后的完整论文 ID 集合
        """
        return self.cache_service.add_paper_ids(scholar_id, new_paper_ids)
    
    def get_cache_stats(self, scholar_id: str) -> Dict[str, Any]:
        """
        获取学者缓存统计
        
        Args:
            scholar_id: Semantic Scholar ID
            
        Returns:
            统计信息字典
        """
        return self.cache_service.get_cache_stats(scholar_id)
    
    def get_all_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有学者的缓存统计
        
        Returns:
            {scholar_id: stats} 字典
        """
        stats = {}
        for scholar in self.list_tracked_scholars():
            stats[scholar.semantic_scholar_id] = {
                "name": scholar.name,
                **self.get_cache_stats(scholar.semantic_scholar_id),
            }
        return stats
    
    def update_scholar_last_checked(self, scholar: Scholar):
        """
        更新学者的最后检查时间
        
        Args:
            scholar: 学者对象
        """
        scholar.last_checked = datetime.now()
    
    def get_settings(self) -> Dict[str, Any]:
        """
        获取配置设置
        
        Returns:
            设置字典
        """
        return self.subscription_service.get_settings()
    
    def get_output_dir(self) -> Path:
        """
        获取报告输出目录
        
        Returns:
            输出目录路径
        """
        return self.subscription_service.get_output_dir()
    
    def clear_scholar_cache(self, scholar_id: str) -> bool:
        """
        清除学者的缓存
        
        Args:
            scholar_id: Semantic Scholar ID
            
        Returns:
            是否成功
        """
        return self.cache_service.clear_cache(scholar_id)
    
    def clear_all_cache(self) -> int:
        """
        清除所有缓存
        
        Returns:
            清除的文件数量
        """
        return self.cache_service.clear_all_cache()
    
    def summary(self) -> str:
        """
        生成学者追踪摘要
        
        Returns:
            摘要文本
        """
        scholars = self.list_tracked_scholars()
        settings = self.get_settings()
        
        lines = [
            "=" * 50,
            "Scholar Tracking Summary",
            "=" * 50,
            f"Total scholars tracked: {len(scholars)}",
            f"Check interval: {settings.get('check_interval')}",
            f"Min influence score: {settings.get('min_influence_score')}",
            "",
            "Tracked Scholars:",
        ]
        
        for scholar in scholars:
            cache_stats = self.get_cache_stats(scholar.semantic_scholar_id)
            lines.append(
                f"  - {scholar.name} (ID: {scholar.semantic_scholar_id})"
                f" | Cached papers: {cache_stats.get('paper_count', 0)}"
                f" | History entries: {cache_stats.get('history_length', 0)}"
            )
        
        lines.append("=" * 50)
        return "\n".join(lines)

    def record_processed_papers(
        self,
        scholar_id: str,
        processed_papers: List[Dict[str, Any]],
    ):
        """记录已处理论文的增量信息"""
        self.cache_service.append_run_history(scholar_id, processed_papers)
