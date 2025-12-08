# src/paperbot/domain/influence/calculator.py
"""
影响力评分计算器
实现 PaperBot Impact Score (PIS) 计算
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .result import InfluenceResult
from .metrics import AcademicMetricsCalculator, EngineeringMetricsCalculator
from .weights import INFLUENCE_WEIGHTS

logger = logging.getLogger(__name__)


class InfluenceCalculator:
    """
    影响力评分计算器
    
    实现 PaperBot Impact Score (PIS) 计算公式:
    Score = w1 * I_a + w2 * I_e
    
    其中:
    - I_a: 学术影响力 (Academic Impact)
    - I_e: 工程影响力 (Engineering Impact)
    - w1, w2: 权重 (默认 0.6, 0.4)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化计算器
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        
        # 权重
        weights = self.config.get("weights", INFLUENCE_WEIGHTS)
        self.w1 = weights.get("academic_weight", 0.6)
        self.w2 = weights.get("engineering_weight", 0.4)
        
        # 初始化子计算器
        self.academic_calc = AcademicMetricsCalculator()
        self.engineering_calc = EngineeringMetricsCalculator()
    
    def calculate(
        self,
        paper,
        code_meta=None,
    ) -> InfluenceResult:
        """
        计算论文的影响力评分
        
        Args:
            paper: 论文元数据 (PaperMeta)
            code_meta: 代码仓库元数据 (CodeMeta, 可选)
            
        Returns:
            影响力评分结果
        """
        # 1. 计算学术影响力
        academic_score = self.academic_calc.compute_score(paper)
        academic_metrics = self.academic_calc.compute(paper)
        
        # 2. 计算工程影响力
        engineering_score = self.engineering_calc.compute_score(paper, code_meta)
        engineering_metrics = self.engineering_calc.compute(paper, code_meta)
        
        # 3. 计算总分
        total_score = self.w1 * academic_score + self.w2 * engineering_score
        recency_factor = self._apply_recency_factor(paper)
        total_score *= recency_factor
        total_score = min(100, max(0, total_score))
        
        # 4. 生成解释
        explanation = self._generate_explanation(
            paper, code_meta,
            academic_score, engineering_score,
            academic_metrics, engineering_metrics,
        )
        
        # 5. 构建结果
        result = InfluenceResult(
            total_score=total_score,
            academic_score=academic_score,
            engineering_score=engineering_score,
            explanation=explanation,
            metrics_breakdown={
                "academic": academic_metrics.to_dict(),
                "engineering": engineering_metrics.to_dict(),
                "weights": {"w1": self.w1, "w2": self.w2, "recency_factor": recency_factor},
            },
        )
        
        title = getattr(paper, 'title', '')[:50]
        logger.info(f"Calculated influence score for '{title}...': {total_score:.1f}")
        
        return result
    
    def _generate_explanation(
        self,
        paper,
        code_meta,
        academic_score: float,
        engineering_score: float,
        academic_metrics,
        engineering_metrics,
    ) -> str:
        """
        生成评分解释
        """
        parts = []
        
        # 学术影响力解释
        academic_explanation = self.academic_calc.explain(paper)
        parts.append(f"**学术影响力** ({academic_score:.1f}/100): {academic_explanation}")
        
        # 工程影响力解释
        engineering_explanation = self.engineering_calc.explain(paper, code_meta)
        parts.append(f"**工程影响力** ({engineering_score:.1f}/100): {engineering_explanation}")
        
        # 总结
        total = self.w1 * academic_score + self.w2 * engineering_score
        parts.append(
            f"\n综合评分公式: {self.w1:.1f} × {academic_score:.1f} + "
            f"{self.w2:.1f} × {engineering_score:.1f} = **{total:.1f}**"
        )
        
        return "\n".join(parts)

    def _apply_recency_factor(self, paper) -> float:
        """
        基于年份的时间衰减
        """
        half_life = self.config.get("weights", {}).get("recency_half_life_years") or INFLUENCE_WEIGHTS.get("recency_half_life_years")
        if not half_life:
            return 1.0
        try:
            year = getattr(paper, 'year', None)
            if year:
                year = int(year)
            else:
                return 1.0
            age = max(0, datetime.now().year - year)
            factor = 0.5 ** (age / float(half_life))
            return max(0.3, min(1.0, factor))
        except Exception:
            return 1.0
    
    def batch_calculate(
        self,
        papers: list,
        code_metas: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, InfluenceResult]:
        """
        批量计算多篇论文的影响力评分
        
        Args:
            papers: 论文元数据列表
            code_metas: 代码仓库元数据字典 {paper_id: CodeMeta}
            
        Returns:
            {paper_id: InfluenceResult} 字典
        """
        code_metas = code_metas or {}
        results = {}
        
        for paper in papers:
            paper_id = getattr(paper, 'paper_id', None)
            code_meta = code_metas.get(paper_id) if paper_id else None
            result = self.calculate(paper, code_meta)
            if paper_id:
                results[paper_id] = result
        
        return results
    
    def get_recommendation(self, score: float) -> str:
        """
        根据分数获取推荐级别
        
        Args:
            score: 影响力分数
            
        Returns:
            推荐级别文本
        """
        if score >= 80:
            return "强烈推荐深入阅读"
        elif score >= 60:
            return "建议关注"
        elif score >= 40:
            return "可选阅读"
        else:
            return "低优先级"
    
    def compare(
        self,
        paper1,
        paper2,
        code_meta1=None,
        code_meta2=None,
    ) -> Dict[str, Any]:
        """
        比较两篇论文的影响力
        
        Args:
            paper1: 论文1
            paper2: 论文2
            code_meta1: 论文1的代码元数据
            code_meta2: 论文2的代码元数据
            
        Returns:
            比较结果字典
        """
        result1 = self.calculate(paper1, code_meta1)
        result2 = self.calculate(paper2, code_meta2)
        
        title1 = getattr(paper1, 'title', 'Paper 1')
        title2 = getattr(paper2, 'title', 'Paper 2')
        
        return {
            "paper1": {
                "title": title1,
                "score": result1.total_score,
                "recommendation": result1.recommendation.value,
            },
            "paper2": {
                "title": title2,
                "score": result2.total_score,
                "recommendation": result2.recommendation.value,
            },
            "difference": result1.total_score - result2.total_score,
            "winner": title1 if result1.total_score > result2.total_score else title2,
        }

