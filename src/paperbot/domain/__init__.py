# src/paperbot/domain/__init__.py
"""
PaperBot 领域模型层。

包含核心业务实体和领域逻辑：
- paper: 论文相关模型
- scholar: 学者相关模型
- influence: 影响力计算
"""

from .paper import PaperMeta, CodeMeta
from .scholar import Scholar
from .influence import (
    InfluenceCalculator,
    InfluenceResult,
    AcademicMetrics,
    EngineeringMetrics,
    AcademicMetricsCalculator,
    EngineeringMetricsCalculator,
    INFLUENCE_WEIGHTS,
)

__all__ = [
    "PaperMeta",
    "CodeMeta",
    "Scholar",
    "InfluenceCalculator",
    "InfluenceResult",
    "AcademicMetrics",
    "EngineeringMetrics",
    "AcademicMetricsCalculator",
    "EngineeringMetricsCalculator",
    "INFLUENCE_WEIGHTS",
]
