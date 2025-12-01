# scholar_tracking/agents/paper_tracker_agent.py
"""
论文追踪 Agent
负责检测学者的新论文并触发分析流水线
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

from agents.base_agent import BaseAgent
from ..models import Scholar, PaperMeta
from .semantic_scholar_agent import SemanticScholarAgent
from .scholar_profile_agent import ScholarProfileAgent

logger = logging.getLogger(__name__)


class PaperTrackerAgent(BaseAgent):
    """
    论文追踪 Agent
    
    主要功能：
    - 检测学者的新发表论文
    - 与缓存对比识别新论文
    - 触发后续分析流水线
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 初始化依赖的 Agent
        self.profile_agent = ScholarProfileAgent(config)
        self.ss_agent = SemanticScholarAgent(config)
        
        # 获取设置
        self.settings = self.profile_agent.get_settings()
        self.papers_per_scholar = self.settings.get("papers_per_scholar", 20)
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        主处理方法：检测所有订阅学者的新论文
        
        Returns:
            包含新论文信息的字典
        """
        results = await self.track_all_scholars()
        return {
            "scholars_checked": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
    
    async def track_all_scholars(self) -> List[Dict[str, Any]]:
        """
        追踪所有订阅学者的新论文
        
        Returns:
            每个学者的追踪结果列表
        """
        scholars = self.profile_agent.list_tracked_scholars()
        results = []
        
        logger.info(f"Starting to track {len(scholars)} scholars")
        
        for scholar in scholars:
            try:
                result = await self.track_scholar(scholar)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to track scholar {scholar.name}: {e}")
                results.append({
                    "scholar_id": scholar.semantic_scholar_id,
                    "scholar_name": scholar.name,
                    "status": "error",
                    "error": str(e),
                    "new_papers": [],
                })
        
        # 关闭 API 客户端
        await self.ss_agent.close()
        
        return results
    
    async def track_scholar(self, scholar: Scholar) -> Dict[str, Any]:
        """
        追踪单个学者的新论文
        
        Args:
            scholar: 学者对象
            
        Returns:
            追踪结果字典
        """
        logger.info(f"Tracking scholar: {scholar.name} (ID: {scholar.semantic_scholar_id})")
        
        # 1. 获取该学者当前的论文列表
        current_papers = await self.ss_agent.fetch_papers_by_author(
            scholar.semantic_scholar_id,
            limit=self.papers_per_scholar,
        )
        
        if not current_papers:
            logger.warning(f"No papers found for {scholar.name}")
            return {
                "scholar_id": scholar.semantic_scholar_id,
                "scholar_name": scholar.name,
                "status": "no_papers",
                "total_papers": 0,
                "new_papers": [],
            }
        
        # 2. 加载缓存的已处理论文 ID
        cached_paper_ids = self.profile_agent.load_scholar_cache(
            scholar.semantic_scholar_id
        )
        
        # 3. 找出新论文
        current_paper_ids = {p.paper_id for p in current_papers}
        new_paper_ids = current_paper_ids - cached_paper_ids
        
        new_papers = [p for p in current_papers if p.paper_id in new_paper_ids]
        
        logger.info(
            f"Scholar {scholar.name}: "
            f"total={len(current_papers)}, cached={len(cached_paper_ids)}, new={len(new_papers)}"
        )
        
        # 4. 更新缓存
        if new_paper_ids:
            self.profile_agent.add_to_cache(
                scholar.semantic_scholar_id,
                new_paper_ids,
            )
        
        # 5. 更新学者的最后检查时间
        self.profile_agent.update_scholar_last_checked(scholar)
        
        return {
            "scholar_id": scholar.semantic_scholar_id,
            "scholar_name": scholar.name,
            "status": "success",
            "total_papers": len(current_papers),
            "cached_papers": len(cached_paper_ids),
            "new_papers_count": len(new_papers),
            "new_papers": [p.to_dict() for p in new_papers],
        }
    
    async def detect_new_papers(
        self,
        scholar: Scholar,
    ) -> Tuple[List[PaperMeta], Set[str]]:
        """
        检测学者的新论文（返回论文对象）
        
        Args:
            scholar: 学者对象
            
        Returns:
            (新论文列表, 所有当前论文ID集合)
        """
        # 获取当前论文
        current_papers = await self.ss_agent.fetch_papers_by_author(
            scholar.semantic_scholar_id,
            limit=self.papers_per_scholar,
        )
        
        # 获取缓存
        cached_paper_ids = self.profile_agent.load_scholar_cache(
            scholar.semantic_scholar_id
        )
        
        # 找出新论文
        current_paper_ids = {p.paper_id for p in current_papers}
        new_paper_ids = current_paper_ids - cached_paper_ids
        
        new_papers = [p for p in current_papers if p.paper_id in new_paper_ids]
        
        return new_papers, current_paper_ids
    
    async def get_tracking_summary(self) -> Dict[str, Any]:
        """
        获取追踪状态摘要
        
        Returns:
            摘要信息字典
        """
        scholars = self.profile_agent.list_tracked_scholars()
        cache_stats = self.profile_agent.get_all_cache_stats()
        
        total_cached = sum(
            stats.get("paper_count", 0) 
            for stats in cache_stats.values()
        )
        
        return {
            "scholars_count": len(scholars),
            "total_cached_papers": total_cached,
            "settings": self.settings,
            "scholars": [
                {
                    "name": s.name,
                    "id": s.semantic_scholar_id,
                    "cached_papers": cache_stats.get(
                        s.semantic_scholar_id, {}
                    ).get("paper_count", 0),
                }
                for s in scholars
            ],
        }
    
    async def force_recheck_scholar(self, scholar_id: str) -> Dict[str, Any]:
        """
        强制重新检查某个学者（清除缓存后重新检测）
        
        Args:
            scholar_id: 学者 ID
            
        Returns:
            检查结果
        """
        # 清除缓存
        self.profile_agent.clear_scholar_cache(scholar_id)
        
        # 获取学者
        scholar = self.profile_agent.get_scholar_by_id(scholar_id)
        if not scholar:
            return {"error": f"Scholar not found: {scholar_id}"}
        
        # 重新追踪
        return await self.track_scholar(scholar)
