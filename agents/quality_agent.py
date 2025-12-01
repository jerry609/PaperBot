# securipaperbot/agents/quality_agent.py

from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path
from .base_agent import BaseAgent


class QualityMetric(Enum):
    CODE_COMPLEXITY = "code_complexity"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TEST_COVERAGE = "test_coverage"


class QualityAgent(BaseAgent):
    """负责代码质量评估的代理"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.metrics = {
            QualityMetric.CODE_COMPLEXITY: self._analyze_complexity,
            QualityMetric.MAINTAINABILITY: self._analyze_maintainability,
            QualityMetric.SECURITY: self._analyze_security,
            QualityMetric.DOCUMENTATION: self._analyze_documentation,
            QualityMetric.TEST_COVERAGE: self._analyze_test_coverage
        }
        self.thresholds = self._load_thresholds()

    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """处理代码质量评估"""
        # 适配 Coordinator 调用：process(context)
        analysis_results = None
        if args and isinstance(args[0], dict):
            context = args[0]
            # 如果上下文中有 code_analysis 结果，使用它
            if "code_analysis" in context and isinstance(context["code_analysis"], dict):
                # 如果是扁平化结果，没法做深入分析，只能返回空或模拟值
                if "analysis_results" not in context["code_analysis"]:
                     return self._process_flat_result(context)
                analysis_results = context["code_analysis"]
            elif "analysis_results" in context:
                analysis_results = context
        elif kwargs and "analysis_results" in kwargs:
            analysis_results = kwargs["analysis_results"]
            
        if not analysis_results:
             # 无法进行深入分析，返回基础结构
             return {
                 'quality_scores': {},
                 'summary': "无法进行深入质量评估（缺少代码分析详情）",
                 'overall_assessment': "暂无代码详情，无法评估。",
                 'strengths': [],
                 'weaknesses': []
             }

        try:
            quality_scores = {}
            for repo_result in analysis_results.get('analysis_results', []):
                quality_scores[repo_result['repo_url']] = await self._evaluate_quality(
                    repo_result['analysis']
                )

            return {
                'quality_scores': quality_scores,
                'summary': await self._generate_quality_summary(quality_scores),
                # 添加 Coordinator 需要的字段
                'overall_assessment': await self._generate_quality_summary(quality_scores),
                'strengths': [], # TODO: 从 scores 中提取
                'weaknesses': []
            }
        except Exception as e:
            self.log_error(e)
            raise

    def _process_flat_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理扁平化的上下文（通常只有元数据）"""
        return {
            'quality_scores': {},
            'summary': "基于元数据的基础评估",
            'overall_assessment': "代码已公开，但未进行深度静态分析。",
            'strengths': ["代码开源"],
            'weaknesses': ["缺少深度质量指标"]
        }

    async def _generate_quality_summary(self, quality_scores: Dict[str, Any]) -> str:
        """生成质量摘要（简单的实现）"""
        if not quality_scores:
            return "没有可用的质量评分。"
        
        summary = []
        for url, score in quality_scores.items():
            if score:
                status = score.get('status', '未知')
                overall = score.get('overall_score', 0)
                summary.append(f"仓库 {url}: 质量{status} ({overall:.2f})")
        
        return "; ".join(summary)

    async def _evaluate_quality(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个仓库的代码质量"""
        try:
            scores = {}
            for metric in QualityMetric:
                analyzer = self.metrics[metric]
                scores[metric.value] = await analyzer(repo_analysis)

            overall_score = self._calculate_overall_score(scores)
            recommendation = self._generate_recommendations(scores)

            return {
                'scores': scores,
                'overall_score': overall_score,
                'recommendations': recommendation,
                'status': self._determine_status(overall_score)
            }
        except Exception as e:
            self.log_error(e)
            return None

    async def _analyze_complexity(self, analysis: Dict[str, Any]) -> float:
        """分析代码复杂度"""
        try:
            complexity_metrics = analysis['structure_analysis'].get('complexity', {})

            # 计算加权分数
            weights = {
                'cyclomatic_complexity': 0.4,
                'cognitive_complexity': 0.3,
                'nesting_depth': 0.3
            }

            score = sum(
                weights[metric] * value
                for metric, value in complexity_metrics.items()
            )

            return min(1.0, max(0.0, 1.0 - score / 10.0))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_maintainability(self, analysis: Dict[str, Any]) -> float:
        """分析可维护性"""
        try:
            maintainability_metrics = analysis['quality_analysis'].get('maintainability', {})

            factors = {
                'code_duplication': 0.3,
                'comment_ratio': 0.2,
                'function_length': 0.2,
                'naming_convention': 0.3
            }

            score = sum(
                factors[metric] * value
                for metric, value in maintainability_metrics.items()
            )

            return min(1.0, max(0.0, score))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_security(self, analysis: Dict[str, Any]) -> float:
        """分析安全性"""
        try:
            security_metrics = analysis['security_analysis']

            # 评估各个安全指标
            weights = {
                'vulnerability_count': 0.4,
                'security_best_practices': 0.3,
                'dependency_security': 0.3
            }

            score = sum(
                weights[metric] * (1.0 - value / 10.0)
                for metric, value in security_metrics.items()
            )

            return min(1.0, max(0.0, score))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_documentation(self, analysis: Dict[str, Any]) -> float:
        """分析文档质量"""
        try:
            doc_metrics = analysis['structure_analysis'].get('documentation', {})

            weights = {
                'docstring_coverage': 0.4,
                'readme_quality': 0.3,
                'api_documentation': 0.3
            }

            score = sum(
                weights[metric] * value
                for metric, value in doc_metrics.items()
            )

            return min(1.0, max(0.0, score))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_test_coverage(self, analysis: Dict[str, Any]) -> float:
        """分析测试覆盖率"""
        try:
            test_metrics = analysis['quality_analysis'].get('testing', {})

            weights = {
                'line_coverage': 0.4,
                'branch_coverage': 0.3,
                'test_quality': 0.3
            }

            score = sum(
                weights[metric] * value
                for metric, value in test_metrics.items()
            )

            return min(1.0, max(0.0, score))
        except Exception as e:
            self.log_error(e)
            return 0.0

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """计算总体质量分数"""
        weights = {
            QualityMetric.CODE_COMPLEXITY.value: 0.25,
            QualityMetric.MAINTAINABILITY.value: 0.25,
            QualityMetric.SECURITY.value: 0.2,
            QualityMetric.DOCUMENTATION.value: 0.15,
            QualityMetric.TEST_COVERAGE.value: 0.15
        }

        return sum(weights[metric] * score for metric, score in scores.items())

    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        for metric, score in scores.items():
            if score < self.thresholds[metric]:
                recommendations.append(
                    self._get_recommendation_for_metric(metric, score)
                )

        return recommendations

    def _determine_status(self, overall_score: float) -> str:
        """确定代码质量状态"""
        if overall_score >= 0.8:
            return "高质量"
        elif overall_score >= 0.6:
            return "可接受"
        else:
            return "需要改进"

    def _load_thresholds(self) -> Dict[str, float]:
        """加载质量阈值"""
        return {
            QualityMetric.CODE_COMPLEXITY.value: 0.7,
            QualityMetric.MAINTAINABILITY.value: 0.7,
            QualityMetric.SECURITY.value: 0.8,
            QualityMetric.DOCUMENTATION.value: 0.6,
            QualityMetric.TEST_COVERAGE.value: 0.7
        }

    def _get_recommendation_for_metric(self, metric: str, score: float) -> str:
        """获取特定指标的改进建议"""
        recommendations = {
            QualityMetric.CODE_COMPLEXITY.value: (
                "建议简化代码结构，减少嵌套深度，拆分复杂函数"
            ),
            QualityMetric.MAINTAINABILITY.value: (
                "提高代码可维护性：添加注释，优化命名，减少代码重复"
            ),
            QualityMetric.SECURITY.value: (
                "加强安全性：更新依赖，修复漏洞，遵循安全最佳实践"
            ),
            QualityMetric.DOCUMENTATION.value: (
                "完善文档：添加文档字符串，更新README，补充API文档"
            ),
            QualityMetric.TEST_COVERAGE.value: (
                "增加测试覆盖率：添加单元测试，集成测试和边界测试"
            )
        }

        return recommendations.get(metric, "一般性改进建议")