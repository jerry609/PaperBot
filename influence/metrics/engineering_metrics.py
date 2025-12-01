# influence/metrics/engineering_metrics.py
"""
工程影响力指标计算器
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from scholar_tracking.models import PaperMeta, CodeMeta
from scholar_tracking.models.influence import EngineeringMetrics
from ..weights import (
    INFLUENCE_WEIGHTS,
    CODE_AVAILABILITY_SCORES,
    get_stars_score,
)

logger = logging.getLogger(__name__)


class EngineeringMetricsCalculator:
    """工程影响力指标计算器"""
    
    def __init__(self):
        """初始化计算器"""
        self.weights = INFLUENCE_WEIGHTS["engineering"]
    
    def compute(
        self,
        paper: PaperMeta,
        code_meta: Optional[CodeMeta] = None,
    ) -> EngineeringMetrics:
        """
        计算论文的工程影响力指标
        
        Args:
            paper: 论文元数据
            code_meta: 代码仓库元数据（可选）
            
        Returns:
            工程影响力指标
        """
        metrics = EngineeringMetrics()
        
        # 1. 代码可用性
        has_code = bool(paper.github_url or paper.has_code or code_meta)
        metrics.has_code = has_code
        metrics.code_availability_score = (
            CODE_AVAILABILITY_SCORES["has_code"] if has_code 
            else CODE_AVAILABILITY_SCORES["no_code"]
        )
        
        # 2. 如果有代码仓库信息
        if code_meta:
            # GitHub Stars 评分
            metrics.repo_stars = code_meta.stars
            metrics.stars_score = get_stars_score(code_meta.stars)
            
            # 可复现性评分（如果有）
            if code_meta.reproducibility_score is not None:
                metrics.reproducibility_score = code_meta.reproducibility_score
            else:
                # 基于仓库信息估算可复现性
                metrics.reproducibility_score = self._estimate_reproducibility(code_meta)
            
            # 活跃度评分
            metrics.is_recently_updated = self._is_recently_updated(code_meta)
            metrics.activity_score = self._compute_activity_score(code_meta)
        else:
            # 没有代码仓库信息
            metrics.repo_stars = 0
            metrics.stars_score = 0
            metrics.reproducibility_score = 0
            metrics.is_recently_updated = False
            metrics.activity_score = 0
        
        return metrics
    
    def _estimate_reproducibility(self, code_meta: CodeMeta) -> float:
        """
        基于仓库信息估算可复现性评分
        
        Args:
            code_meta: 代码仓库元数据
            
        Returns:
            可复现性评分 (0-100)
        """
        score = 0
        
        # 有 README
        if code_meta.has_readme:
            score += 30
        
        # 有文档
        if code_meta.has_docs:
            score += 20
        
        # 有 License
        if code_meta.license:
            score += 10
        
        # 根据 Star 数间接判断（高 Star 通常意味着易用）
        if code_meta.stars >= 100:
            score += 20
        elif code_meta.stars >= 10:
            score += 10
        
        # 最近有更新
        if self._is_recently_updated(code_meta):
            score += 20
        
        return min(100, score)
    
    def _is_recently_updated(
        self,
        code_meta: CodeMeta,
        days_threshold: int = 180,
    ) -> bool:
        """
        判断仓库是否最近有更新
        
        Args:
            code_meta: 代码仓库元数据
            days_threshold: 天数阈值
            
        Returns:
            是否最近更新
        """
        if not code_meta.updated_at and not code_meta.last_commit_date:
            return False
        
        try:
            last_update_str = code_meta.updated_at or code_meta.last_commit_date
            if not last_update_str:
                return False
            
            # 解析日期
            if "T" in last_update_str:
                last_update = datetime.fromisoformat(
                    last_update_str.replace("Z", "+00:00")
                )
            else:
                last_update = datetime.strptime(last_update_str, "%Y-%m-%d")
            
            # 比较
            threshold = datetime.now(last_update.tzinfo) - timedelta(days=days_threshold)
            return last_update > threshold
        except Exception as e:
            logger.warning(f"Failed to parse date: {e}")
            return False
    
    def _compute_activity_score(self, code_meta: CodeMeta) -> float:
        """
        计算活跃度评分
        
        Args:
            code_meta: 代码仓库元数据
            
        Returns:
            活跃度评分 (0-100)
        """
        score = 0
        
        # 最近更新
        if self._is_recently_updated(code_meta, 30):
            score += 40  # 30天内更新
        elif self._is_recently_updated(code_meta, 90):
            score += 30  # 90天内更新
        elif self._is_recently_updated(code_meta, 180):
            score += 20  # 180天内更新
        elif self._is_recently_updated(code_meta, 365):
            score += 10  # 一年内更新
        
        # Forks 数
        if code_meta.forks >= 100:
            score += 30
        elif code_meta.forks >= 10:
            score += 20
        elif code_meta.forks >= 1:
            score += 10
        
        # Open Issues (适量的 issue 说明社区活跃)
        if 1 <= code_meta.open_issues <= 50:
            score += 20
        elif code_meta.open_issues > 50:
            score += 10
        
        return min(100, score)
    
    def compute_score(
        self,
        paper: PaperMeta,
        code_meta: Optional[CodeMeta] = None,
    ) -> float:
        """
        计算论文的工程影响力总分
        
        Args:
            paper: 论文元数据
            code_meta: 代码仓库元数据（可选）
            
        Returns:
            工程影响力分数 (0-100)
        """
        metrics = self.compute(paper, code_meta)
        
        # 加权计算
        code_weight = self.weights.get("code_availability_weight", 0.3)
        stars_weight = self.weights.get("stars_weight", 0.4)
        repro_weight = self.weights.get("reproducibility_weight", 0.3)
        
        score = (
            metrics.code_availability_score * code_weight +
            metrics.stars_score * stars_weight +
            metrics.reproducibility_score * repro_weight
        )
        
        return min(100, max(0, score))
    
    def explain(
        self,
        paper: PaperMeta,
        code_meta: Optional[CodeMeta] = None,
    ) -> str:
        """
        生成工程影响力评分解释
        
        Args:
            paper: 论文元数据
            code_meta: 代码仓库元数据（可选）
            
        Returns:
            解释文本
        """
        metrics = self.compute(paper, code_meta)
        
        parts = []
        
        # 代码可用性
        if metrics.has_code:
            parts.append("代码公开可用")
        else:
            parts.append("未发现公开代码")
        
        # GitHub Stars
        if metrics.repo_stars > 0:
            parts.append(f"GitHub Stars: {metrics.repo_stars}")
        
        # 活跃度
        if metrics.is_recently_updated:
            parts.append("仓库近期活跃")
        
        return "; ".join(parts) if parts else "无工程信息"
