# scholar_tracking/models/influence.py
"""
影响力评分数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class RecommendationLevel(Enum):
    """推荐级别枚举"""
    HIGHLY_RECOMMENDED = "强烈推荐深入阅读"
    RECOMMENDED = "建议关注"
    OPTIONAL = "可选阅读"
    LOW_PRIORITY = "低优先级"


@dataclass
class InfluenceResult:
    """影响力评分结果模型"""
    
    # 总分 (0-100)
    total_score: float
    
    # 学术影响力分数 (Academic Impact, I_a)
    academic_score: float
    
    # 工程影响力分数 (Engineering Impact, I_e)
    engineering_score: float
    
    # 评分解释
    explanation: str = ""
    
    # 各项细分指标
    metrics_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    # 推荐级别
    recommendation: RecommendationLevel = RecommendationLevel.OPTIONAL
    
    # 可选：趋势影响力 (未来扩展)
    trend_score: Optional[float] = None
    
    def __post_init__(self):
        """后处理：确定推荐级别"""
        if self.total_score >= 80:
            self.recommendation = RecommendationLevel.HIGHLY_RECOMMENDED
        elif self.total_score >= 60:
            self.recommendation = RecommendationLevel.RECOMMENDED
        elif self.total_score >= 40:
            self.recommendation = RecommendationLevel.OPTIONAL
        else:
            self.recommendation = RecommendationLevel.LOW_PRIORITY
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_score": round(self.total_score, 2),
            "academic_score": round(self.academic_score, 2),
            "engineering_score": round(self.engineering_score, 2),
            "trend_score": round(self.trend_score, 2) if self.trend_score else None,
            "explanation": self.explanation,
            "metrics_breakdown": self.metrics_breakdown,
            "recommendation": self.recommendation.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfluenceResult":
        """从字典创建"""
        # 解析推荐级别
        recommendation = RecommendationLevel.OPTIONAL
        rec_value = data.get("recommendation", "")
        for level in RecommendationLevel:
            if level.value == rec_value:
                recommendation = level
                break
        
        result = cls(
            total_score=data["total_score"],
            academic_score=data["academic_score"],
            engineering_score=data["engineering_score"],
            explanation=data.get("explanation", ""),
            metrics_breakdown=data.get("metrics_breakdown", {}),
            trend_score=data.get("trend_score"),
        )
        result.recommendation = recommendation
        return result
    
    def get_summary(self) -> str:
        """获取评分摘要"""
        return (
            f"影响力评分: {self.total_score:.1f}/100 "
            f"(学术: {self.academic_score:.1f}, 工程: {self.engineering_score:.1f}) - "
            f"{self.recommendation.value}"
        )
    
    def __str__(self) -> str:
        return self.get_summary()
    
    def __repr__(self) -> str:
        return f"InfluenceResult(total={self.total_score:.1f}, academic={self.academic_score:.1f}, eng={self.engineering_score:.1f})"


@dataclass
class AcademicMetrics:
    """学术影响力细分指标"""
    
    # 引用相关
    citation_count: int = 0
    citation_score: float = 0.0  # 映射后的分数
    
    # 顶会相关
    is_top_venue: bool = False
    venue_tier: Optional[int] = None  # 1 = tier1, 2 = tier2, None = other
    venue_score: float = 0.0
    
    # 作者影响力 (可选，后续扩展)
    author_h_index: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_count": self.citation_count,
            "citation_score": self.citation_score,
            "is_top_venue": self.is_top_venue,
            "venue_tier": self.venue_tier,
            "venue_score": self.venue_score,
            "author_h_index": self.author_h_index,
        }


@dataclass
class EngineeringMetrics:
    """工程影响力细分指标"""
    
    # 代码可用性
    has_code: bool = False
    code_availability_score: float = 0.0
    
    # GitHub 指标
    repo_stars: int = 0
    stars_score: float = 0.0
    
    # 可复现性 (由 CodeAnalysisAgent 评估)
    reproducibility_score: float = 0.0
    
    # 活跃度
    is_recently_updated: bool = False
    activity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_code": self.has_code,
            "code_availability_score": self.code_availability_score,
            "repo_stars": self.repo_stars,
            "stars_score": self.stars_score,
            "reproducibility_score": self.reproducibility_score,
            "is_recently_updated": self.is_recently_updated,
            "activity_score": self.activity_score,
        }
