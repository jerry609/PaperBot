# securipaperbot/utils/analyzer.py

from typing import Dict, List, Any, Optional
from pathlib import Path
import ast
import re
import radon.complexity as radon
from radon.visitors import ComplexityVisitor
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from paperbot.utils.logger import setup_logger


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    complexity_score: float
    maintainability_index: float
    security_score: float
    documentation_score: float
    test_coverage: float
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class CodeAnalyzer:
    """代码分析工具类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_logger(__name__)
        self.thread_local = threading.local()

        # 安全漏洞模式
        self.security_patterns = {
            'sql_injection': r'.*exec\s*\(.*\$.*\).*',
            'xss': r'.*innerHTML\s*=.*',
            'command_injection': r'.*eval\s*\(.*\).*',
            'path_traversal': r'.*\.\.\/.*',
        }

        # 代码质量指标权重
        self.quality_weights = {
            'complexity': 0.3,
            'maintainability': 0.25,
            'security': 0.25,
            'documentation': 0.1,
            'test_coverage': 0.1
        }

    async def analyze_structure(self, repo_path: Path) -> Dict[str, Any]:
        """分析代码结构"""
        try:
            structure = {
                'files': self._analyze_files(repo_path),
                'dependencies': self._analyze_dependencies(repo_path),
                'complexity': self._analyze_code_complexity(repo_path),
                'documentation': self._analyze_documentation(repo_path)
            }
            return structure
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {str(e)}")
            return {}

    async def analyze_security(self, repo_path: Path) -> Dict[str, Any]:
        """分析安全性"""
        try:
            security_report = {
                'vulnerabilities': await self._find_vulnerabilities(repo_path),
                'security_measures': self._analyze_security_measures(repo_path),
                'dependency_security': await self._check_dependency_security(repo_path)
            }
            return security_report
        except Exception as e:
            self.logger.error(f"Security analysis failed: {str(e)}")
            return {}

    def _analyze_files(self, repo_path: Path) -> Dict[str, Any]:
        """分析文件结构"""
        file_stats = {
            'total_files': 0,
            'file_types': {},
            'size_distribution': {},
            'file_structure': {}
        }

        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                file_stats['total_files'] += 1

                # 统计文件类型
                ext = file_path.suffix
                file_stats['file_types'][ext] = file_stats['file_types'].get(ext, 0) + 1

                # 统计文件大小分布
                size = file_path.stat().st_size
                size_category = self._categorize_file_size(size)
                file_stats['size_distribution'][size_category] = \
                    file_stats['size_distribution'].get(size_category, 0) + 1

                # 构建文件结构树
                relative_path = file_path.relative_to(repo_path)
                self._update_file_structure(file_stats['file_structure'], relative_path)

        return file_stats

    def _analyze_dependencies(self, repo_path: Path) -> Dict[str, Any]:
        """分析项目依赖"""
        dependencies = {
            'direct_dependencies': {},
            'dev_dependencies': {},
            'dependency_graph': {},
            'outdated_dependencies': []
        }

        # 检查各种依赖文件
        dependency_files = {
            'requirements.txt': self._parse_python_requirements,
            'package.json': self._parse_node_dependencies,
            'Cargo.toml': self._parse_rust_dependencies,
            'pom.xml': self._parse_maven_dependencies
        }

        for dep_file, parser in dependency_files.items():
            dep_path = repo_path / dep_file
            if dep_path.exists():
                dependencies.update(parser(dep_path))

        return dependencies

    def _analyze_code_complexity(self, repo_path: Path) -> Dict[str, Any]:
        """分析代码复杂度"""
        complexity_metrics = {
            'overall_complexity': 0,
            'file_complexity': {},
            'function_complexity': {},
            'complexity_distribution': {}
        }

        python_files = list(repo_path.rglob('*.py'))

        with ThreadPoolExecutor() as executor:
            file_results = list(executor.map(self._analyze_file_complexity, python_files))

        for file_path, result in zip(python_files, file_results):
            relative_path = str(file_path.relative_to(repo_path))
            complexity_metrics['file_complexity'][relative_path] = result
            complexity_metrics['overall_complexity'] += result['total_complexity']

            # 更新复杂度分布
            complexity_level = self._categorize_complexity(result['total_complexity'])
            complexity_metrics['complexity_distribution'][complexity_level] = \
                complexity_metrics['complexity_distribution'].get(complexity_level, 0) + 1

        return complexity_metrics

    def _analyze_documentation(self, repo_path: Path) -> Dict[str, Any]:
        """分析文档质量"""
        doc_metrics = {
            'docstring_coverage': 0,
            'documentation_files': [],
            'documentation_quality': {},
            'api_documentation': {}
        }

        # 分析Python文件的文档字符串
        python_files = list(repo_path.rglob('*.py'))
        total_functions = 0
        documented_functions = 0

        for file_path in python_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                except Exception as e:
                    self.logger.warning(f"Failed to parse {file_path}: {str(e)}")

        if total_functions > 0:
            doc_metrics['docstring_coverage'] = documented_functions / total_functions

        # 检查文档文件
        doc_patterns = ['*.md', '*.rst', '*.txt']
        for pattern in doc_patterns:
            doc_files = list(repo_path.rglob(pattern))
            for doc_file in doc_files:
                doc_metrics['documentation_files'].append({
                    'path': str(doc_file.relative_to(repo_path)),
                    'size': doc_file.stat().st_size,
                    'quality_score': self._assess_doc_quality(doc_file)
                })

        return doc_metrics

    async def _find_vulnerabilities(self, repo_path: Path) -> List[Dict[str, Any]]:
        """查找潜在的安全漏洞"""
        vulnerabilities = []

        # 遍历所有代码文件
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.php']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查安全模式
                    for vuln_type, pattern in self.security_patterns.items():
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            vulnerabilities.append({
                                'type': vuln_type,
                                'file': str(file_path.relative_to(repo_path)),
                                'line': content.count('\n', 0, match.start()) + 1,
                                'snippet': match.group(0),
                                'severity': self._assess_vulnerability_severity(vuln_type)
                            })
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {file_path}: {str(e)}")

        return vulnerabilities

    def _analyze_security_measures(self, repo_path: Path) -> Dict[str, Any]:
        """分析安全措施"""
        security_measures = {
            'input_validation': self._check_input_validation(repo_path),
            'authentication': self._check_authentication_mechanisms(repo_path),
            'encryption': self._check_encryption_usage(repo_path),
            'secure_headers': self._check_secure_headers(repo_path),
            'csrf_protection': self._check_csrf_protection(repo_path)
        }
        return security_measures

    async def _check_dependency_security(self, repo_path: Path) -> Dict[str, Any]:
        """检查依赖的安全性"""
        try:
            # 使用safety检查Python依赖
            result = subprocess.run(
                ['safety', 'check', '-r', 'requirements.txt'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            vulnerabilities = []
            if result.returncode != 0:
                for line in result.stdout.splitlines():
                    if 'Found vulnerability' in line:
                        vulnerabilities.append(self._parse_safety_output(line))

            return {
                'vulnerable_dependencies': vulnerabilities,
                'total_vulnerabilities': len(vulnerabilities),
                'scan_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Dependency security check failed: {str(e)}")
            return {}

    def _categorize_file_size(self, size: int) -> str:
        """对文件大小进行分类"""
        if size < 1024:  # 1KB
            return 'tiny'
        elif size < 10240:  # 10KB
            return 'small'
        elif size < 102400:  # 100KB
            return 'medium'
        elif size < 1024000:  # 1MB
            return 'large'
        else:
            return 'huge'

    def _categorize_complexity(self, complexity: int) -> str:
        """对复杂度进行分类"""
        if complexity <= 5:
            return 'simple'
        elif complexity <= 10:
            return 'moderate'
        elif complexity <= 20:
            return 'complex'
        else:
            return 'very_complex'

    def _assess_vulnerability_severity(self, vuln_type: str) -> str:
        """评估漏洞严重程度"""
        severity_levels = {
            'sql_injection': 'critical',
            'command_injection': 'critical',
            'xss': 'high',
            'path_traversal': 'medium'
        }
        return severity_levels.get(vuln_type, 'low')