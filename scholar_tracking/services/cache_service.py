# scholar_tracking/services/cache_service.py
"""
缓存管理服务
用于存储和检索已处理的论文 ID
"""

import json
import logging
from pathlib import Path
from typing import Set, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheService:
    """论文缓存管理服务"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初始化缓存服务
        
        Args:
            cache_dir: 缓存目录路径
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "cache" / "scholar_papers"
        
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, scholar_id: str) -> Path:
        """获取学者的缓存文件路径"""
        return self.cache_dir / f"{scholar_id}.json"
    
    def load_paper_ids(self, scholar_id: str) -> Set[str]:
        """
        加载学者的已处理论文 ID 集合
        
        Args:
            scholar_id: 学者的 Semantic Scholar ID
            
        Returns:
            已处理的论文 ID 集合
        """
        cache_path = self._get_cache_path(scholar_id)
        
        if not cache_path.exists():
            logger.debug(f"No cache found for scholar {scholar_id}")
            return set()
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            paper_ids = set(data.get("paper_ids", []))
            logger.debug(f"Loaded {len(paper_ids)} cached paper IDs for scholar {scholar_id}")
            return paper_ids
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid cache file for {scholar_id}: {e}")
            return set()
        except Exception as e:
            logger.error(f"Failed to load cache for {scholar_id}: {e}")
            return set()
    
    def save_paper_ids(self, scholar_id: str, paper_ids: Set[str]):
        """
        保存学者的已处理论文 ID 集合
        
        Args:
            scholar_id: 学者的 Semantic Scholar ID
            paper_ids: 论文 ID 集合
        """
        cache_path = self._get_cache_path(scholar_id)
        
        data = {
            "scholar_id": scholar_id,
            "paper_ids": list(paper_ids),
            "count": len(paper_ids),
            "last_updated": datetime.now().isoformat(),
        }
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {len(paper_ids)} paper IDs for scholar {scholar_id}")
        except Exception as e:
            logger.error(f"Failed to save cache for {scholar_id}: {e}")
            raise
    
    def add_paper_ids(self, scholar_id: str, new_paper_ids: Set[str]) -> Set[str]:
        """
        添加新的论文 ID 到缓存
        
        Args:
            scholar_id: 学者的 Semantic Scholar ID
            new_paper_ids: 新的论文 ID 集合
            
        Returns:
            更新后的完整论文 ID 集合
        """
        existing = self.load_paper_ids(scholar_id)
        updated = existing | new_paper_ids
        self.save_paper_ids(scholar_id, updated)
        return updated
    
    def get_cache_stats(self, scholar_id: str) -> Dict[str, Any]:
        """
        获取学者缓存的统计信息
        
        Args:
            scholar_id: 学者的 Semantic Scholar ID
            
        Returns:
            缓存统计信息字典
        """
        cache_path = self._get_cache_path(scholar_id)
        
        if not cache_path.exists():
            return {
                "exists": False,
                "paper_count": 0,
                "last_updated": None,
            }
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return {
                "exists": True,
                "paper_count": data.get("count", len(data.get("paper_ids", []))),
                "last_updated": data.get("last_updated"),
            }
        except Exception:
            return {
                "exists": True,
                "paper_count": 0,
                "last_updated": None,
                "error": True,
            }
    
    def clear_cache(self, scholar_id: str) -> bool:
        """
        清除学者的缓存
        
        Args:
            scholar_id: 学者的 Semantic Scholar ID
            
        Returns:
            是否成功清除
        """
        cache_path = self._get_cache_path(scholar_id)
        
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Cleared cache for scholar {scholar_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache for {scholar_id}: {e}")
                return False
        return True
    
    def clear_all_cache(self) -> int:
        """
        清除所有缓存
        
        Returns:
            清除的缓存文件数量
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete {cache_file}: {e}")
        
        logger.info(f"Cleared {count} cache files")
        return count
