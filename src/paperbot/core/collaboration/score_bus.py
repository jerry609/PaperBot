# src/paperbot/core/collaboration/score_bus.py
"""
评分共享总线 (Score Share Bus)

P3 增强:
- 允许后续阶段访问前序评分
- 支持评分订阅机制
- 支持阈值判断
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger(__name__)


@dataclass
class StageScore:
    """
    阶段评分数据
    
    Attributes:
        stage: 阶段名称 (research/code/quality/influence)
        score: 评分 (0-100)
        confidence: 置信度 (0-1)
        key_metrics: 关键指标字典
        timestamp: 评分时间
    """
    stage: str
    score: float
    confidence: float = 1.0
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "score": self.score,
            "confidence": self.confidence,
            "key_metrics": self.key_metrics,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @property
    def weighted_score(self) -> float:
        """置信度加权评分"""
        return self.score * self.confidence


class ScoreShareBus:
    """
    评分共享总线
    
    功能:
    - 发布/订阅评分更新
    - 允许后续阶段根据前序评分决策
    - 提供聚合评分计算
    
    使用场景:
    - 低质量论文跳过深度分析
    - 无代码论文跳过代码分析
    - 空壳仓库降低整体评分
    """
    
    def __init__(self, paper_id: Optional[str] = None):
        """
        初始化评分总线
        
        Args:
            paper_id: 论文ID,用于跟踪
        """
        self.paper_id = paper_id or ""
        self._scores: Dict[str, StageScore] = {}
        self._subscribers: List[Callable[[StageScore], None]] = []
        self._history: List[StageScore] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def publish_score(self, score: StageScore) -> None:
        """
        发布阶段评分
        
        Args:
            score: 阶段评分数据
        """
        self._scores[score.stage] = score
        self._history.append(score)
        
        self.logger.info(
            f"[{self.paper_id}] Stage '{score.stage}' score: {score.score:.1f} "
            f"(confidence: {score.confidence:.2f})"
        )
        
        # 通知订阅者
        for callback in self._subscribers:
            try:
                callback(score)
            except Exception as e:
                self.logger.warning(f"Subscriber callback failed: {e}")
    
    def get_score(self, stage: str) -> Optional[StageScore]:
        """
        获取指定阶段的评分
        
        Args:
            stage: 阶段名称
            
        Returns:
            StageScore 或 None
        """
        return self._scores.get(stage)
    
    def get_all_scores(self) -> Dict[str, StageScore]:
        """获取所有阶段评分"""
        return dict(self._scores)
    
    def subscribe(self, callback: Callable[[StageScore], None]) -> None:
        """
        订阅评分更新
        
        Args:
            callback: 回调函数,接收 StageScore 参数
        """
        self._subscribers.append(callback)
    
    def get_aggregate_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        计算加权聚合评分
        
        Args:
            weights: 阶段权重字典,默认等权
            
        Returns:
            聚合评分 (0-100)
        """
        if not self._scores:
            return 0.0
        
        if weights is None:
            # 默认等权
            weights = {stage: 1.0 for stage in self._scores}
        
        total_weight = sum(weights.get(s, 0) for s in self._scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            self._scores[s].weighted_score * weights.get(s, 1.0)
            for s in self._scores if s in weights
        )
        
        return weighted_sum / total_weight
    
    def meets_threshold(self, stage: str, threshold: float) -> bool:
        """
        检查阶段评分是否达到阈值
        
        Args:
            stage: 阶段名称
            threshold: 阈值
            
        Returns:
            True 如果评分 >= 阈值,或阶段不存在
        """
        score = self._scores.get(stage)
        if score is None:
            return True  # 不存在视为通过
        return score.score >= threshold
    
    def get_lowest_score(self) -> Optional[StageScore]:
        """获取最低评分阶段"""
        if not self._scores:
            return None
        return min(self._scores.values(), key=lambda s: s.score)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "paper_id": self.paper_id,
            "scores": {k: v.to_dict() for k, v in self._scores.items()},
            "aggregate_score": self.get_aggregate_score(),
        }
    
    def clear(self) -> None:
        """清空评分"""
        self._scores.clear()
        self._history.clear()


# ==================== 便捷工厂函数 ====================

def create_research_score(
    citation_count: int,
    venue_tier: Optional[int],
    confidence: float = 1.0,
) -> StageScore:
    """创建研究阶段评分"""
    # 简单评分逻辑
    score = min(100, citation_count / 10)  # 1000+ citations = 100
    if venue_tier == 1:
        score = max(score, 60)  # 顶会至少60分
    elif venue_tier == 2:
        score = max(score, 40)
    
    return StageScore(
        stage="research",
        score=score,
        confidence=confidence,
        key_metrics={
            "citation_count": citation_count,
            "venue_tier": venue_tier,
        }
    )


def create_code_score(
    has_code: bool,
    health_score: float = 0.0,
    is_empty_repo: bool = False,
    confidence: float = 1.0,
) -> StageScore:
    """创建代码分析阶段评分"""
    if not has_code:
        score = 0.0
    elif is_empty_repo:
        score = 10.0  # 空壳仓库低分
    else:
        score = health_score
    
    return StageScore(
        stage="code",
        score=score,
        confidence=confidence,
        key_metrics={
            "has_code": has_code,
            "health_score": health_score,
            "is_empty_repo": is_empty_repo,
        }
    )


def create_quality_score(
    quality_score: float,
    maintainability: float = 0.0,
    test_coverage: float = 0.0,
    confidence: float = 1.0,
) -> StageScore:
    """创建质量评估阶段评分"""
    return StageScore(
        stage="quality",
        score=quality_score,
        confidence=confidence,
        key_metrics={
            "maintainability": maintainability,
            "test_coverage": test_coverage,
        }
    )


def create_influence_score(
    total_score: float,
    academic_score: float,
    engineering_score: float,
    momentum_score: float = 0.0,
    confidence: float = 1.0,
) -> StageScore:
    """创建影响力阶段评分"""
    return StageScore(
        stage="influence",
        score=total_score,
        confidence=confidence,
        key_metrics={
            "academic_score": academic_score,
            "engineering_score": engineering_score,
            "momentum_score": momentum_score,
        }
    )
