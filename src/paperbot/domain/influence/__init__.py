# src/paperbot/domain/influence/__init__.py
"""
影响力计算模块。

提供论文和学者影响力评估功能：
- InfluenceCalculator: 影响力计算器
- AcademicMetricsCalculator: 学术影响力计算
- EngineeringMetricsCalculator: 工程影响力计算
"""

from .result import InfluenceResult, AcademicMetrics, EngineeringMetrics
from .calculator import InfluenceCalculator
from .weights import INFLUENCE_WEIGHTS, get_citation_score, get_stars_score
from .metrics import AcademicMetricsCalculator, EngineeringMetricsCalculator

__all__ = [
    "InfluenceCalculator",
    "InfluenceResult",
    "AcademicMetrics",
    "EngineeringMetrics",
    "AcademicMetricsCalculator",
    "EngineeringMetricsCalculator",
    "INFLUENCE_WEIGHTS",
    "get_citation_score",
    "get_stars_score",
]

