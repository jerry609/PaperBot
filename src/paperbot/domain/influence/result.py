# src/paperbot/domain/influence/result.py
"""
影响力评估结果模型。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class RecommendationLevel(str, Enum):
    """推荐级别。"""
    HIGHLY_RECOMMENDED = "强烈推荐深入阅读"
    RECOMMENDED = "建议关注"
    OPTIONAL = "可选阅读"
    LOW_PRIORITY = "低优先级"


@dataclass
class AcademicMetrics:
    """学术影响力指标。"""
    citation_count: int = 0
    citation_score: float = 0.0
    venue_tier: Optional[int] = None
    venue_score: float = 0.0
    is_top_venue: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "citation_count": self.citation_count,
            "citation_score": self.citation_score,
            "venue_tier": self.venue_tier,
            "venue_score": self.venue_score,
            "is_top_venue": self.is_top_venue,
        }


@dataclass
class EngineeringMetrics:
    """工程影响力指标。"""
    has_code: bool = False
    code_availability_score: float = 0.0
    repo_stars: int = 0
    stars_score: float = 0.0
    reproducibility_score: float = 0.0
    is_recently_updated: bool = False
    activity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "has_code": self.has_code,
            "code_availability_score": self.code_availability_score,
            "repo_stars": self.repo_stars,
            "stars_score": self.stars_score,
            "reproducibility_score": self.reproducibility_score,
            "is_recently_updated": self.is_recently_updated,
            "activity_score": self.activity_score,
        }


@dataclass
class InfluenceResult:
    """影响力评估结果。"""
    total_score: float = 0.0
    academic_score: float = 0.0
    engineering_score: float = 0.0
    explanation: str = ""
    metrics_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def recommendation(self) -> RecommendationLevel:
        """获取推荐级别。"""
        if self.total_score >= 80:
            return RecommendationLevel.HIGHLY_RECOMMENDED
        elif self.total_score >= 60:
            return RecommendationLevel.RECOMMENDED
        elif self.total_score >= 40:
            return RecommendationLevel.OPTIONAL
        else:
            return RecommendationLevel.LOW_PRIORITY
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "total_score": self.total_score,
            "academic_score": self.academic_score,
            "engineering_score": self.engineering_score,
            "recommendation": self.recommendation.value,
            "explanation": self.explanation,
            "metrics_breakdown": self.metrics_breakdown,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfluenceResult":
        """从字典创建实例。"""
        return cls(
            total_score=data.get("total_score", 0.0),
            academic_score=data.get("academic_score", 0.0),
            engineering_score=data.get("engineering_score", 0.0),
            explanation=data.get("explanation", ""),
            metrics_breakdown=data.get("metrics_breakdown", {}),
        )
