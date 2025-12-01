# influence/metrics/academic_metrics.py
"""
学术影响力指标计算器
"""

import logging
from typing import Dict, Any, Optional, Set
from pathlib import Path
import yaml

from scholar_tracking.models import PaperMeta
from scholar_tracking.models.influence import AcademicMetrics
from ..weights import (
    INFLUENCE_WEIGHTS,
    VENUE_SCORES,
    get_citation_score,
)

logger = logging.getLogger(__name__)


class AcademicMetricsCalculator:
    """学术影响力指标计算器"""
    
    def __init__(self, venues_config_path: Optional[Path] = None):
        """
        初始化计算器
        
        Args:
            venues_config_path: 顶会配置文件路径
        """
        self.weights = INFLUENCE_WEIGHTS["academic"]
        
        # 加载顶会配置
        if venues_config_path is None:
            project_root = Path(__file__).parent.parent.parent
            venues_config_path = project_root / "config" / "top_venues.yaml"
        
        self.tier1_venues: Set[str] = set()
        self.tier2_venues: Set[str] = set()
        self._load_venues_config(venues_config_path)
    
    def _load_venues_config(self, config_path: Path):
        """加载顶会配置"""
        if not config_path.exists():
            logger.warning(f"Venues config not found: {config_path}")
            return
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # 遍历所有领域
            for field, tiers in config.items():
                if field == "scoring":
                    continue
                if isinstance(tiers, dict):
                    if "tier1" in tiers:
                        self.tier1_venues.update(
                            v.lower() for v in tiers["tier1"]
                        )
                    if "tier2" in tiers:
                        self.tier2_venues.update(
                            v.lower() for v in tiers["tier2"]
                        )
            
            logger.info(
                f"Loaded venues config: {len(self.tier1_venues)} tier1, "
                f"{len(self.tier2_venues)} tier2"
            )
        except Exception as e:
            logger.error(f"Failed to load venues config: {e}")
    
    def compute(self, paper: PaperMeta) -> AcademicMetrics:
        """
        计算论文的学术影响力指标
        
        Args:
            paper: 论文元数据
            
        Returns:
            学术影响力指标
        """
        metrics = AcademicMetrics()
        
        # 1. 引用数评分
        metrics.citation_count = paper.citation_count
        metrics.citation_score = get_citation_score(paper.citation_count)
        
        # 2. 发表渠道评分
        venue_tier = self._get_venue_tier(paper.venue)
        metrics.venue_tier = venue_tier
        metrics.is_top_venue = venue_tier == 1
        
        if venue_tier == 1:
            metrics.venue_score = VENUE_SCORES["tier1"]
        elif venue_tier == 2:
            metrics.venue_score = VENUE_SCORES["tier2"]
        else:
            metrics.venue_score = VENUE_SCORES["other"]
        
        return metrics
    
    def _get_venue_tier(self, venue: Optional[str]) -> Optional[int]:
        """
        判断发表渠道的级别
        
        Args:
            venue: 发表渠道名称
            
        Returns:
            1 = tier1, 2 = tier2, None = other
        """
        if not venue:
            return None
        
        venue_lower = venue.lower()
        
        # 检查 tier1
        for t1 in self.tier1_venues:
            if t1 in venue_lower or venue_lower in t1:
                return 1
        
        # 检查 tier2
        for t2 in self.tier2_venues:
            if t2 in venue_lower or venue_lower in t2:
                return 2
        
        return None
    
    def compute_score(self, paper: PaperMeta) -> float:
        """
        计算论文的学术影响力总分
        
        Args:
            paper: 论文元数据
            
        Returns:
            学术影响力分数 (0-100)
        """
        metrics = self.compute(paper)
        
        # 加权计算
        citation_weight = self.weights.get("citation_weight", 0.6)
        venue_weight = self.weights.get("venue_weight", 0.4)
        
        score = (
            metrics.citation_score * citation_weight +
            metrics.venue_score * venue_weight
        )
        
        return min(100, max(0, score))
    
    def explain(self, paper: PaperMeta) -> str:
        """
        生成学术影响力评分解释
        
        Args:
            paper: 论文元数据
            
        Returns:
            解释文本
        """
        metrics = self.compute(paper)
        
        parts = []
        
        # 引用数解释
        parts.append(f"引用数: {metrics.citation_count}")
        
        # 发表渠道解释
        venue_str = paper.venue or "未知"
        if metrics.venue_tier == 1:
            parts.append(f"发表于顶会: {venue_str}")
        elif metrics.venue_tier == 2:
            parts.append(f"发表于优秀会议: {venue_str}")
        else:
            parts.append(f"发表于: {venue_str}")
        
        return "; ".join(parts)
