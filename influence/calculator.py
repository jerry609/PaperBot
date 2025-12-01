# influence/calculator.py
"""
影响力评分计算器
实现 PaperBot Impact Score (PIS) 计算
"""

import logging
from typing import Dict, Any, Optional

from scholar_tracking.models import PaperMeta, CodeMeta
from scholar_tracking.models.influence import InfluenceResult
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
        paper: PaperMeta,
        code_meta: Optional[CodeMeta] = None,
    ) -> InfluenceResult:
        """
        计算论文的影响力评分
        
        Args:
            paper: 论文元数据
            code_meta: 代码仓库元数据（可选）
            
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
                "weights": {"w1": self.w1, "w2": self.w2},
            },
        )
        
        logger.info(f"Calculated influence score for '{paper.title[:50]}...': {total_score:.1f}")
        
        return result
    
    def _generate_explanation(
        self,
        paper: PaperMeta,
        code_meta: Optional[CodeMeta],
        academic_score: float,
        engineering_score: float,
        academic_metrics,
        engineering_metrics,
    ) -> str:
        """
        生成评分解释
        
        Args:
            paper: 论文元数据
            code_meta: 代码仓库元数据
            academic_score: 学术影响力分数
            engineering_score: 工程影响力分数
            academic_metrics: 学术指标
            engineering_metrics: 工程指标
            
        Returns:
            解释文本
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
    
    def batch_calculate(
        self,
        papers: list,
        code_metas: Optional[Dict[str, CodeMeta]] = None,
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
            code_meta = code_metas.get(paper.paper_id)
            result = self.calculate(paper, code_meta)
            results[paper.paper_id] = result
        
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
        paper1: PaperMeta,
        paper2: PaperMeta,
        code_meta1: Optional[CodeMeta] = None,
        code_meta2: Optional[CodeMeta] = None,
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
        
        return {
            "paper1": {
                "title": paper1.title,
                "score": result1.total_score,
                "recommendation": result1.recommendation.value,
            },
            "paper2": {
                "title": paper2.title,
                "score": result2.total_score,
                "recommendation": result2.recommendation.value,
            },
            "difference": result1.total_score - result2.total_score,
            "winner": paper1.title if result1.total_score > result2.total_score else paper2.title,
        }
