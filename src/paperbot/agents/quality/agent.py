# src/paperbot/agents/quality/agent.py
"""
负责代码质量评估的代理
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from ..base import BaseAgent


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

    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """执行质量评估"""
        return await self.process(*args, **kwargs)

    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """处理代码质量评估"""
        analysis_input = self._extract_analysis_input(*args, **kwargs)
        if not analysis_input:
            return {
                'quality_score': 0.0,
                'quality_scores': {},
                'summary': "无法进行深入质量评估（缺少代码分析详情）",
                'overall_assessment': "暂无代码详情，无法评估。",
                'strengths': [],
                'weaknesses': [],
            }

        if self._is_flat_result(analysis_input):
            return self._process_flat_result(analysis_input)

        try:
            quality_scores = {}
            repo_entries = self._normalize_repo_entries(analysis_input)
            for repo_result in repo_entries:
                repo_url = repo_result.get('repo_url') or repo_result.get('repo_name') or 'unknown'
                if repo_result.get('placeholder'):
                    quality_scores[repo_url] = self._placeholder_quality(repo_result)
                    continue

                analysis = repo_result.get('analysis', repo_result)
                quality_scores[repo_url] = await self._evaluate_quality(analysis)

            overall_values = [
                score['overall_score']
                for score in quality_scores.values()
                if isinstance(score, dict)
            ]
            overall_score = (
                sum(overall_values) / len(overall_values)
                if overall_values
                else 0.0
            )

            return {
                'quality_score': overall_score,
                'quality_scores': quality_scores,
                'summary': await self._generate_quality_summary(quality_scores),
                'overall_assessment': await self._generate_quality_summary(quality_scores),
                'strengths': self._collect_strengths(quality_scores),
                'weaknesses': self._collect_weaknesses(quality_scores),
            }
        except Exception as e:
            self.log_error(e)
            raise

    def _process_flat_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理扁平化的上下文（通常只有元数据）"""
        raw_score = context.get('reproducibility_score')
        quality_score = float(raw_score or 0.0)
        if quality_score > 1.0:
            quality_score /= 100.0

        strengths = []
        if context.get('has_readme'):
            strengths.append("仓库包含 README")
        if raw_score is not None:
            strengths.append(f"代码复现度信号: {float(raw_score):.0f}/100")

        weaknesses = []
        if context.get('placeholder'):
            weaknesses.append(f"仓库分析不可用: {context.get('reason', 'unknown')}")
        else:
            weaknesses.append("当前仅有仓库元数据，缺少静态分析细节")

        return {
            'quality_score': quality_score,
            'quality_scores': {},
            'summary': "基于元数据的基础评估",
            'overall_assessment': (
                "代码仓库已发现，但当前链路只拿到了元数据，尚未建立完整静态分析画像。"
            ),
            'strengths': strengths or ["代码开源"],
            'weaknesses': weaknesses,
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

    async def _evaluate_quality(self, repo_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            quality_metrics = analysis.get('quality_analysis', {})
            if 'complexity_score' in quality_metrics:
                return min(1.0, max(0.0, float(quality_metrics.get('complexity_score', 0.0))))

            complexity_metrics = analysis.get('structure_analysis', {}).get('complexity', {})
            total_complexity = float(complexity_metrics.get('overall_complexity', 0.0) or 0.0)
            file_count = max(1, len(complexity_metrics.get('file_complexity', {})))
            average_complexity = total_complexity / file_count
            return min(1.0, max(0.0, 1.0 - (average_complexity / 20.0)))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_maintainability(self, analysis: Dict[str, Any]) -> float:
        """分析可维护性"""
        try:
            quality_metrics = analysis.get('quality_analysis', {})
            if 'maintainability_score' in quality_metrics:
                return min(1.0, max(0.0, float(quality_metrics.get('maintainability_score', 0.0))))

            overall = float(quality_metrics.get('overall_score', 0.0) or 0.0)
            documentation = float(quality_metrics.get('documentation_score', 0.0) or 0.0)
            complexity = float(quality_metrics.get('complexity_score', 0.0) or 0.0)
            has_readme = 1.0 if quality_metrics.get('has_readme') else 0.0
            score = (overall * 0.5) + (documentation * 0.2) + (complexity * 0.2) + (has_readme * 0.1)
            return min(1.0, max(0.0, score))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_security(self, analysis: Dict[str, Any]) -> float:
        """分析安全性"""
        try:
            security_metrics = analysis.get('security_analysis', {})
            vulnerabilities = security_metrics.get('vulnerabilities', []) or []
            dependency_security = security_metrics.get('dependency_security', {}) or {}
            dependency_vulns = float(dependency_security.get('total_vulnerabilities', 0.0) or 0.0)

            vulnerability_score = max(0.0, 1.0 - ((len(vulnerabilities) + dependency_vulns) / 5.0))
            measures = security_metrics.get('security_measures', {}) or {}
            coverage = [
                1.0 if self._security_measure_present(value) else 0.0
                for value in measures.values()
            ]
            measures_score = sum(coverage) / len(coverage) if coverage else 0.0
            score = (vulnerability_score * 0.7) + (measures_score * 0.3)
            return min(1.0, max(0.0, score))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_documentation(self, analysis: Dict[str, Any]) -> float:
        """分析文档质量"""
        try:
            doc_metrics = analysis.get('structure_analysis', {}).get('documentation', {})
            docstring_coverage = float(doc_metrics.get('docstring_coverage', 0.0) or 0.0)
            readme_quality = float(doc_metrics.get('readme_quality', 0.0) or 0.0)
            api_docs = doc_metrics.get('api_documentation', {}) or {}
            api_coverage = float(api_docs.get('coverage', 0.0) or 0.0)
            score = (
                (docstring_coverage * 0.5)
                + (readme_quality * 0.3)
                + (api_coverage * 0.2)
            )
            return min(1.0, max(0.0, score))
        except Exception as e:
            self.log_error(e)
            return 0.0

    async def _analyze_test_coverage(self, analysis: Dict[str, Any]) -> float:
        """分析测试覆盖率"""
        try:
            quality_metrics = analysis.get('quality_analysis', {})
            if 'test_coverage_score' in quality_metrics:
                return min(1.0, max(0.0, float(quality_metrics.get('test_coverage_score', 0.0))))

            files = analysis.get('structure_analysis', {}).get('files', {})
            file_paths = files.get('file_paths', [])
            if not file_paths:
                return 0.0

            test_count = sum(1 for path in file_paths if self._is_test_file(path))
            code_count = sum(
                1 for path in file_paths if self._is_source_file(path) and not self._is_test_file(path)
            )
            if not code_count:
                return 0.0

            ratio = test_count / code_count
            return min(1.0, max(0.0, ratio * 2.0))
        except Exception as e:
            self.log_error(e)
            return 0.0

    def _is_test_file(self, path: str) -> bool:
        return (
            path.startswith('tests/')
            or '/tests/' in path
            or path.endswith('_test.py')
            or path.endswith('.spec.ts')
            or path.endswith('.test.ts')
            or path.endswith('.test.tsx')
            or path.endswith('.spec.js')
            or path.endswith('.test.js')
        )

    def _is_source_file(self, path: str) -> bool:
        return path.endswith(('.py', '.ts', '.tsx', '.js', '.jsx'))

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

    def _extract_analysis_input(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        if args and isinstance(args[0], dict):
            context = args[0]
            for key in ('code_analysis_result', 'code_analysis', 'analysis_results'):
                value = context.get(key)
                if isinstance(value, dict):
                    return value
            if any(key in context for key in ('structure_analysis', 'repo_url', 'placeholder')):
                return context

        for key in ('code_analysis_result', 'analysis_results'):
            value = kwargs.get(key)
            if isinstance(value, dict):
                return value

        return None

    def _is_flat_result(self, analysis_input: Dict[str, Any]) -> bool:
        if 'analysis_results' in analysis_input or 'structure_analysis' in analysis_input:
            return False
        return any(
            key in analysis_input
            for key in ('repo_url', 'repo_name', 'reproducibility_score', 'placeholder')
        )

    def _normalize_repo_entries(self, analysis_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        if 'analysis_results' in analysis_input:
            entries = analysis_input.get('analysis_results') or []
            return [entry for entry in entries if isinstance(entry, dict)]

        if 'structure_analysis' in analysis_input:
            return [{'repo_url': analysis_input.get('repo_url', 'unknown'), 'analysis': analysis_input}]

        return []

    def _placeholder_quality(self, repo_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'scores': {
                QualityMetric.CODE_COMPLEXITY.value: 0.0,
                QualityMetric.MAINTAINABILITY.value: 0.0,
                QualityMetric.SECURITY.value: 0.0,
                QualityMetric.DOCUMENTATION.value: 0.0,
                QualityMetric.TEST_COVERAGE.value: 0.0,
            },
            'overall_score': 0.0,
            'recommendations': [f"Repository unavailable: {repo_result.get('reason', 'unknown')}"],
            'status': "需要改进",
        }

    def _collect_strengths(self, quality_scores: Dict[str, Any]) -> List[str]:
        strengths: List[str] = []
        for score in quality_scores.values():
            if not isinstance(score, dict):
                continue
            metric_scores = score.get('scores', {})
            if metric_scores.get(QualityMetric.DOCUMENTATION.value, 0.0) >= 0.7:
                strengths.append("文档完整度较好")
            if metric_scores.get(QualityMetric.SECURITY.value, 0.0) >= 0.7:
                strengths.append("安全基线较稳定")
            if metric_scores.get(QualityMetric.TEST_COVERAGE.value, 0.0) >= 0.6:
                strengths.append("测试信号较强")

        return list(dict.fromkeys(strengths))

    def _collect_weaknesses(self, quality_scores: Dict[str, Any]) -> List[str]:
        weaknesses: List[str] = []
        for score in quality_scores.values():
            if not isinstance(score, dict):
                continue
            weaknesses.extend(score.get('recommendations', []))
        return list(dict.fromkeys(weaknesses))

    def _security_measure_present(self, value: Any) -> bool:
        if isinstance(value, dict):
            if value.get('present'):
                return True
            matches = value.get('matches')
            return isinstance(matches, list) and bool(matches)
        return bool(value)

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
