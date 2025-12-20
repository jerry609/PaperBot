# src/paperbot/domain/influence/result.py
"""
影响力评估结果模型。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class RecommendationLevel(str, Enum):
    """推荐级别。"""
    HIGHLY_RECOMMENDED = "强烈推荐深入阅读"
    RECOMMENDED = "建议关注"
    OPTIONAL = "可选阅读"
    LOW_PRIORITY = "低优先级"


class CitationTrend(str, Enum):
    """引用增长趋势。"""
    ACCELERATING = "accelerating"  # 加速增长
    STABLE = "stable"              # 稳定
    DECLINING = "declining"        # 下降


class CitationSentiment(str, Enum):
    """引用情感类型。"""
    POSITIVE = "positive"   # 正面引用: 扩展、验证
    NEGATIVE = "negative"   # 负面引用: 批评、反驳
    NEUTRAL = "neutral"     # 中性引用: 简单引用


class DependencyRiskLevel(str, Enum):
    """依赖风险等级。"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AcademicMetrics:
    """学术影响力指标。"""
    citation_count: int = 0
    citation_score: float = 0.0
    venue_tier: Optional[int] = None
    venue_score: float = 0.0
    is_top_venue: bool = False
    # P2 新增: 引用动态指标
    citation_velocity: Optional["CitationVelocity"] = None
    momentum_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {
            "citation_count": self.citation_count,
            "citation_score": self.citation_score,
            "venue_tier": self.venue_tier,
            "venue_score": self.venue_score,
            "is_top_venue": self.is_top_venue,
            "momentum_score": self.momentum_score,
        }
        if self.citation_velocity:
            result["citation_velocity"] = self.citation_velocity.to_dict()
        return result


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
    # P2 新增: 代码健康指标
    health: Optional["CodeHealthResult"] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {
            "has_code": self.has_code,
            "code_availability_score": self.code_availability_score,
            "repo_stars": self.repo_stars,
            "stars_score": self.stars_score,
            "reproducibility_score": self.reproducibility_score,
            "is_recently_updated": self.is_recently_updated,
            "activity_score": self.activity_score,
        }
        if self.health:
            result["health"] = self.health.to_dict()
        return result


@dataclass
class InfluenceResult:
    """影响力评估结果。"""
    total_score: float = 0.0
    academic_score: float = 0.0
    engineering_score: float = 0.0
    explanation: str = ""
    metrics_breakdown: Dict[str, Any] = field(default_factory=dict)
    # P2 新增: 情感分析结果
    sentiment_result: Optional["CitationSentimentResult"] = None
    
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
        result = {
            "total_score": self.total_score,
            "academic_score": self.academic_score,
            "engineering_score": self.engineering_score,
            "recommendation": self.recommendation.value,
            "explanation": self.explanation,
            "metrics_breakdown": self.metrics_breakdown,
        }
        if self.sentiment_result:
            result["sentiment"] = self.sentiment_result.to_dict()
        return result
    
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


# ==================== P2 新增数据模型 ====================

@dataclass
class CitationVelocity:
    """引用增速指标。"""
    recent_citations: int = 0          # 近期引用数 (默认6个月)
    annual_average: float = 0.0        # 年均引用
    growth_rate: float = 0.0           # 增长率 (%)
    trend: CitationTrend = CitationTrend.STABLE
    window_months: int = 6             # 统计窗口
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recent_citations": self.recent_citations,
            "annual_average": self.annual_average,
            "growth_rate": self.growth_rate,
            "trend": self.trend.value,
            "window_months": self.window_months,
        }


@dataclass
class CitationContext:
    """单条引用的语境信息。"""
    citing_paper_id: str
    citing_paper_title: str
    context_text: str
    sentiment: CitationSentiment = CitationSentiment.NEUTRAL
    confidence: float = 0.0
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "citing_paper_id": self.citing_paper_id,
            "citing_paper_title": self.citing_paper_title,
            "context_text": self.context_text[:200],
            "sentiment": self.sentiment.value,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class CitationSentimentResult:
    """引用情感分析结果。"""
    total_analyzed: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    sentiment_score: float = 50.0      # 0-100, 50 = neutral
    notable_critiques: List[str] = field(default_factory=list)
    contexts: List[CitationContext] = field(default_factory=list)
    
    @property
    def positive_ratio(self) -> float:
        if self.total_analyzed == 0:
            return 0.0
        return self.positive_count / self.total_analyzed
    
    @property
    def negative_ratio(self) -> float:
        if self.total_analyzed == 0:
            return 0.0
        return self.negative_count / self.total_analyzed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_analyzed": self.total_analyzed,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "positive_ratio": round(self.positive_ratio, 2),
            "negative_ratio": round(self.negative_ratio, 2),
            "sentiment_score": self.sentiment_score,
            "notable_critiques": self.notable_critiques[:3],
        }


@dataclass
class DependencyRisk:
    """依赖风险项。"""
    package: str
    risk_type: str                     # "outdated" | "vulnerable" | "deprecated"
    severity: DependencyRiskLevel = DependencyRiskLevel.LOW
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "package": self.package,
            "risk_type": self.risk_type,
            "severity": self.severity.value,
            "description": self.description,
        }


@dataclass
class CodeHealthResult:
    """代码健康检查结果。"""
    health_score: float = 0.0          # 0-100
    is_empty_repo: bool = False        # 空壳仓库
    doc_coverage: float = 0.0          # 文档覆盖率 0-100%
    has_readme: bool = False
    has_docs_folder: bool = False
    has_tests: bool = False
    commit_count: int = 0
    dependency_risks: List[DependencyRisk] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_score": self.health_score,
            "is_empty_repo": self.is_empty_repo,
            "doc_coverage": self.doc_coverage,
            "has_readme": self.has_readme,
            "has_docs_folder": self.has_docs_folder,
            "has_tests": self.has_tests,
            "commit_count": self.commit_count,
            "dependency_risks": [r.to_dict() for r in self.dependency_risks],
            "warnings": self.warnings,
        }

