# paperbot/utils/analyzer.py

import ast
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

try:
    import radon.complexity as radon
    from radon.visitors import ComplexityVisitor
except ImportError:
    radon = None
    ComplexityVisitor = None

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
        structure = self._empty_structure()
        try:
            files = self._analyze_files(repo_path)
            documentation = self._analyze_documentation(repo_path)
            structure = {
                'files': files,
                'dependencies': self._analyze_dependencies(repo_path),
                'complexity': self._analyze_code_complexity(repo_path),
                'documentation': documentation,
                'primary_language': self._detect_primary_language(files.get('file_types', {})),
            }
            return structure
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {str(e)}")
            return structure

    async def analyze_security(self, repo_path: Path) -> Dict[str, Any]:
        """分析安全性"""
        security_report = self._empty_security_report()
        try:
            security_report = {
                'vulnerabilities': await self._find_vulnerabilities(repo_path),
                'security_measures': self._analyze_security_measures(repo_path),
                'dependency_security': await self._check_dependency_security(repo_path)
            }
            return security_report
        except Exception as e:
            self.logger.error(f"Security analysis failed: {str(e)}")
            return security_report

    async def analyze_quality(self, repo_path: Path) -> Dict[str, Any]:
        """分析代码质量"""
        try:
            complexity = self._analyze_code_complexity(repo_path)
            documentation = self._analyze_documentation(repo_path)
            files = self._analyze_files(repo_path)
            
            # 计算总体质量分数
            overall_score = 0.0
            recommendations = []
            
            # 复杂度评分 (0-1)
            total_complexity = complexity.get('overall_complexity', 0)
            file_count = max(1, len(complexity.get('file_complexity', {})))
            average_complexity = total_complexity / file_count
            complexity_score = max(0.0, 1.0 - (average_complexity / 20.0))
            overall_score += complexity_score * self.quality_weights['complexity']
            
            if complexity_score < 0.5:
                recommendations.append("Consider refactoring complex functions")
            
            # 文档评分
            doc_coverage = documentation.get('docstring_coverage', 0)
            overall_score += doc_coverage * self.quality_weights['documentation']
            
            if doc_coverage < 0.5:
                recommendations.append("Improve documentation coverage")
            
            # 检查是否有 README
            has_readme = documentation.get('has_readme', False)
            if not has_readme:
                recommendations.append("Add a repository README")

            test_coverage_score = self._estimate_test_signal(files)
            overall_score += test_coverage_score * self.quality_weights['test_coverage']
            if test_coverage_score < 0.4:
                recommendations.append("Add or expand automated tests")

            maintainability_score = min(
                1.0,
                (complexity_score * 0.6)
                + (doc_coverage * 0.2)
                + ((1.0 if has_readme else 0.0) * 0.2),
            )
            overall_score += maintainability_score * self.quality_weights['maintainability']
            
            return {
                'overall_score': min(1.0, overall_score),
                'complexity_score': complexity_score,
                'maintainability_score': maintainability_score,
                'documentation_score': doc_coverage,
                'test_coverage_score': test_coverage_score,
                'has_readme': has_readme,
                'recommendations': recommendations,
                'complexity_metrics': complexity,
                'documentation_metrics': documentation,
            }
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {str(e)}")
            return {
                'overall_score': 0.0,
                'complexity_score': 0.0,
                'maintainability_score': 0.0,
                'documentation_score': 0.0,
                'test_coverage_score': 0.0,
                'has_readme': False,
                'recommendations': [],
                'complexity_metrics': self._empty_complexity_metrics(),
                'documentation_metrics': self._empty_documentation_metrics(),
            }

    async def analyze_dependencies(self, repo_path: Path) -> Dict[str, Any]:
        """分析项目依赖（异步版本）"""
        try:
            return self._analyze_dependencies(repo_path)
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}")
            return self._empty_dependency_metrics()

    def _analyze_files(self, repo_path: Path) -> Dict[str, Any]:
        """分析文件结构"""
        file_stats = {
            'total_files': 0,
            'file_types': {},
            'size_distribution': {},
            'file_structure': {},
            'file_paths': [],
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
                file_stats['file_paths'].append(str(relative_path))

        file_stats['file_paths'].sort()
        return file_stats

    def _analyze_dependencies(self, repo_path: Path) -> Dict[str, Any]:
        """分析项目依赖"""
        dependencies = self._empty_dependency_metrics()

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
                parsed = parser(dep_path)
                self._merge_dependency_metrics(dependencies, parsed)

        return dependencies

    def _analyze_code_complexity(self, repo_path: Path) -> Dict[str, Any]:
        """分析代码复杂度"""
        complexity_metrics = self._empty_complexity_metrics()

        python_files = list(repo_path.rglob('*.py'))

        with ThreadPoolExecutor() as executor:
            file_results = list(executor.map(self._analyze_file_complexity, python_files))

        for file_path, result in zip(python_files, file_results):
            relative_path = str(file_path.relative_to(repo_path))
            complexity_metrics['file_complexity'][relative_path] = result
            complexity_metrics['overall_complexity'] += result['total_complexity']
            complexity_metrics['function_complexity'][relative_path] = result.get('functions', {})

            # 更新复杂度分布
            complexity_level = self._categorize_complexity(result['total_complexity'])
            complexity_metrics['complexity_distribution'][complexity_level] = \
                complexity_metrics['complexity_distribution'].get(complexity_level, 0) + 1

        return complexity_metrics

    def _analyze_documentation(self, repo_path: Path) -> Dict[str, Any]:
        """分析文档质量"""
        doc_metrics = self._empty_documentation_metrics()

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
        doc_metrics['api_documentation'] = {
            'documented_symbols': documented_functions,
            'total_symbols': total_functions,
            'coverage': doc_metrics['docstring_coverage'],
        }

        # 检查文档文件
        doc_patterns = ['*.md', '*.rst', '*.txt']
        for pattern in doc_patterns:
            doc_files = list(repo_path.rglob(pattern))
            for doc_file in doc_files:
                quality_score = self._assess_doc_quality(doc_file)
                relative_path = str(doc_file.relative_to(repo_path))
                doc_metrics['documentation_files'].append({
                    'path': relative_path,
                    'size': doc_file.stat().st_size,
                    'quality_score': quality_score,
                })
                doc_metrics['documentation_quality'][relative_path] = quality_score
                if doc_file.name.lower().startswith('readme'):
                    doc_metrics['has_readme'] = True
                    doc_metrics['readme_quality'] = max(
                        doc_metrics['readme_quality'],
                        quality_score,
                    )

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
        report = {
            'vulnerable_dependencies': [],
            'total_vulnerabilities': 0,
            'scan_timestamp': datetime.now().isoformat(),
            'status': 'skipped',
            'scanner': 'safety',
        }

        requirements_path = repo_path / 'requirements.txt'
        if not requirements_path.exists():
            return report

        if shutil.which('safety') is None:
            report['status'] = 'unavailable'
            return report

        try:
            # 使用safety检查Python依赖
            result = subprocess.run(
                ['safety', 'check', '-r', 'requirements.txt'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            vulnerabilities = []
            if result.returncode != 0:
                output = "\n".join([result.stdout, result.stderr])
                for line in output.splitlines():
                    if 'Found vulnerability' in line:
                        vulnerabilities.append(self._parse_safety_output(line))

            report['vulnerable_dependencies'] = vulnerabilities
            report['total_vulnerabilities'] = len(vulnerabilities)
            report['status'] = 'issues_found' if vulnerabilities else 'clean'
            return report
        except Exception as e:
            self.logger.error(f"Dependency security check failed: {str(e)}")
            report['status'] = 'error'
            report['scan_error'] = str(e)
            return report

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

    def _empty_structure(self) -> Dict[str, Any]:
        return {
            'files': {
                'total_files': 0,
                'file_types': {},
                'size_distribution': {},
                'file_structure': {},
                'file_paths': [],
            },
            'dependencies': self._empty_dependency_metrics(),
            'complexity': self._empty_complexity_metrics(),
            'documentation': self._empty_documentation_metrics(),
            'primary_language': 'Unknown',
        }

    def _empty_security_report(self) -> Dict[str, Any]:
        return {
            'vulnerabilities': [],
            'security_measures': {
                'input_validation': self._empty_security_measure(),
                'authentication': self._empty_security_measure(),
                'encryption': self._empty_security_measure(),
                'secure_headers': self._empty_security_measure(),
                'csrf_protection': self._empty_security_measure(),
            },
            'dependency_security': {
                'vulnerable_dependencies': [],
                'total_vulnerabilities': 0,
                'scan_timestamp': datetime.now().isoformat(),
                'status': 'skipped',
                'scanner': 'safety',
            },
        }

    def _empty_dependency_metrics(self) -> Dict[str, Any]:
        return {
            'direct_dependencies': {},
            'dev_dependencies': {},
            'dependency_graph': {},
            'outdated_dependencies': [],
        }

    def _empty_complexity_metrics(self) -> Dict[str, Any]:
        return {
            'overall_complexity': 0,
            'file_complexity': {},
            'function_complexity': {},
            'complexity_distribution': {},
        }

    def _empty_documentation_metrics(self) -> Dict[str, Any]:
        return {
            'docstring_coverage': 0.0,
            'documentation_files': [],
            'documentation_quality': {},
            'api_documentation': {
                'documented_symbols': 0,
                'total_symbols': 0,
                'coverage': 0.0,
            },
            'has_readme': False,
            'readme_quality': 0.0,
        }

    def _empty_security_measure(self) -> Dict[str, Any]:
        return {'present': False, 'matches': []}

    def _detect_primary_language(self, file_types: Dict[str, int]) -> str:
        language_map = {
            '.py': 'Python',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.java': 'Java',
            '.rs': 'Rust',
            '.go': 'Go',
            '.php': 'PHP',
        }
        ranked = [
            (language_map.get(ext, ext or 'unknown'), count)
            for ext, count in file_types.items()
            if ext in language_map
        ]
        if not ranked:
            return 'Unknown'
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[0][0]

    def _update_file_structure(self, file_structure: Dict[str, Any], relative_path: Path) -> None:
        current = file_structure
        parts = list(relative_path.parts)
        for index, part in enumerate(parts):
            is_leaf = index == len(parts) - 1
            if is_leaf:
                current.setdefault(part, {})
            else:
                current = current.setdefault(part, {})

    def _merge_dependency_metrics(
        self,
        target: Dict[str, Any],
        parsed: Dict[str, Any],
    ) -> None:
        for section in ('direct_dependencies', 'dev_dependencies', 'dependency_graph'):
            values = parsed.get(section, {})
            if not isinstance(values, dict):
                continue
            for ecosystem, deps in values.items():
                target[section].setdefault(ecosystem, [])
                target[section][ecosystem].extend(deps)

        outdated = parsed.get('outdated_dependencies', [])
        if isinstance(outdated, list):
            target['outdated_dependencies'].extend(outdated)

    def _parse_python_requirements(self, dep_path: Path) -> Dict[str, Any]:
        dependencies = []
        for raw_line in dep_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            name, version = self._split_dependency_spec(line)
            dependencies.append({'name': name, 'version': version})

        names = [dep['name'] for dep in dependencies]
        return {
            'direct_dependencies': {'python': dependencies},
            'dev_dependencies': {},
            'dependency_graph': {'python': names},
            'outdated_dependencies': [],
        }

    def _parse_node_dependencies(self, dep_path: Path) -> Dict[str, Any]:
        try:
            data = json.loads(dep_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return self._empty_dependency_metrics()

        dependencies = [
            {'name': name, 'version': version}
            for name, version in (data.get('dependencies') or {}).items()
        ]
        dev_dependencies = [
            {'name': name, 'version': version}
            for name, version in (data.get('devDependencies') or {}).items()
        ]
        return {
            'direct_dependencies': {'node': dependencies},
            'dev_dependencies': {'node': dev_dependencies},
            'dependency_graph': {
                'node': [dep['name'] for dep in dependencies + dev_dependencies],
            },
            'outdated_dependencies': [],
        }

    def _parse_rust_dependencies(self, dep_path: Path) -> Dict[str, Any]:
        if tomllib is None:
            return self._empty_dependency_metrics()

        data = tomllib.loads(dep_path.read_text(encoding='utf-8'))
        dependencies = [
            {'name': name, 'version': self._normalize_dependency_version(version)}
            for name, version in (data.get('dependencies') or {}).items()
        ]
        dev_dependencies = [
            {'name': name, 'version': self._normalize_dependency_version(version)}
            for name, version in (data.get('dev-dependencies') or {}).items()
        ]
        return {
            'direct_dependencies': {'rust': dependencies},
            'dev_dependencies': {'rust': dev_dependencies},
            'dependency_graph': {
                'rust': [dep['name'] for dep in dependencies + dev_dependencies],
            },
            'outdated_dependencies': [],
        }

    def _parse_maven_dependencies(self, dep_path: Path) -> Dict[str, Any]:
        dependencies = []
        dev_dependencies = []

        try:
            tree = ET.parse(dep_path)
        except ET.ParseError:
            return self._empty_dependency_metrics()

        for dependency in tree.iterfind('.//{*}dependency'):
            group_id = self._xml_text(dependency, 'groupId')
            artifact_id = self._xml_text(dependency, 'artifactId')
            version = self._xml_text(dependency, 'version')
            scope = (self._xml_text(dependency, 'scope') or '').lower()
            entry = {
                'name': f"{group_id}:{artifact_id}".strip(':'),
                'version': version,
            }
            if scope == 'test':
                dev_dependencies.append(entry)
            else:
                dependencies.append(entry)

        return {
            'direct_dependencies': {'maven': dependencies},
            'dev_dependencies': {'maven': dev_dependencies},
            'dependency_graph': {
                'maven': [dep['name'] for dep in dependencies + dev_dependencies],
            },
            'outdated_dependencies': [],
        }

    def _analyze_file_complexity(self, file_path: Path) -> Dict[str, Any]:
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except Exception:
            return {
                'total_complexity': 0,
                'average_complexity': 0.0,
                'max_complexity': 0,
                'functions': {},
            }

        decision_nodes = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.BoolOp,
            ast.With,
            ast.AsyncWith,
            ast.comprehension,
        )
        match_node = getattr(ast, 'Match', None)
        if match_node is not None:
            decision_nodes = decision_nodes + (match_node,)

        functions: Dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                branch_points = sum(
                    1
                    for child in ast.walk(node)
                    if isinstance(
                        child,
                        decision_nodes,
                    )
                )
                functions[node.name] = 1 + branch_points

        total_complexity = sum(functions.values())
        average_complexity = total_complexity / max(1, len(functions))
        return {
            'total_complexity': total_complexity,
            'average_complexity': average_complexity,
            'max_complexity': max(functions.values(), default=0),
            'functions': functions,
        }

    def _assess_doc_quality(self, doc_file: Path) -> float:
        try:
            content = doc_file.read_text(encoding='utf-8')
        except Exception:
            return 0.0

        words = len(re.findall(r'\w+', content))
        headings = len(re.findall(r'^\s*#+\s+', content, re.MULTILINE))
        code_blocks = content.count('```')
        score = 0.0
        if words >= 50:
            score += 0.4
        elif words >= 15:
            score += 0.2
        if headings > 0:
            score += 0.3
        if code_blocks > 0:
            score += 0.3
        return round(min(1.0, score), 2)

    def _check_input_validation(self, repo_path: Path) -> Dict[str, Any]:
        return self._scan_security_pattern(
            repo_path,
            [
                r'\b(BaseModel|pydantic|validator|validate\(|marshmallow|joi|zod)\b',
            ],
        )

    def _check_authentication_mechanisms(self, repo_path: Path) -> Dict[str, Any]:
        return self._scan_security_pattern(
            repo_path,
            [
                r'\b(jwt|oauth|nextauth|authenticate|authorization|bearer)\b',
            ],
        )

    def _check_encryption_usage(self, repo_path: Path) -> Dict[str, Any]:
        return self._scan_security_pattern(
            repo_path,
            [
                r'\b(hashlib|bcrypt|argon2|cryptography|fernet|ssl|tls)\b',
            ],
        )

    def _check_secure_headers(self, repo_path: Path) -> Dict[str, Any]:
        return self._scan_security_pattern(
            repo_path,
            [
                r'Content-Security-Policy',
                r'X-Frame-Options',
                r'Strict-Transport-Security',
                r'helmet\(',
            ],
        )

    def _check_csrf_protection(self, repo_path: Path) -> Dict[str, Any]:
        return self._scan_security_pattern(
            repo_path,
            [
                r'\bcsrf\b',
                r'\bxsrf\b',
            ],
        )

    def _parse_safety_output(self, line: str) -> Dict[str, Any]:
        package_match = re.search(r'vulnerability in ([A-Za-z0-9_.-]+)', line, re.IGNORECASE)
        severity_match = re.search(r'\b(critical|high|medium|low)\b', line, re.IGNORECASE)
        return {
            'package': package_match.group(1) if package_match else 'unknown',
            'severity': severity_match.group(1).lower() if severity_match else 'unknown',
            'advisory': line.strip(),
        }

    def _estimate_test_signal(self, file_stats: Dict[str, Any]) -> float:
        file_paths = file_stats.get('file_paths', [])
        if not file_paths:
            return 0.0

        test_files = [
            path for path in file_paths
            if path.startswith('tests/')
            or '/tests/' in path
            or path.endswith('_test.py')
            or path.endswith('.spec.ts')
            or path.endswith('.test.ts')
            or path.endswith('.test.tsx')
            or path.endswith('.spec.js')
            or path.endswith('.test.js')
        ]
        code_files = [
            path for path in file_paths
            if path.endswith(('.py', '.ts', '.tsx', '.js', '.jsx'))
            and path not in test_files
        ]
        if not code_files:
            return 0.0

        ratio = len(test_files) / max(1, len(code_files))
        return round(min(1.0, ratio * 2.0), 2)

    def _split_dependency_spec(self, spec: str) -> tuple[str, Optional[str]]:
        normalized = spec.strip()
        for operator in ('==', '>=', '<=', '~=', '!=', '>', '<'):
            if operator not in normalized:
                continue
            name, version = normalized.split(operator, 1)
            return name.strip(), version.strip() or None
        return normalized, None

    def _normalize_dependency_version(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            version = value.get('version')
            return str(version) if version else None
        return None

    def _xml_text(self, dependency: ET.Element, tag_name: str) -> Optional[str]:
        for child in dependency:
            if child.tag.endswith(tag_name):
                return (child.text or '').strip() or None
        return None

    def _scan_security_pattern(
        self,
        repo_path: Path,
        patterns: List[str],
    ) -> Dict[str, Any]:
        matches: List[Dict[str, Any]] = []
        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix not in {'.py', '.js', '.ts', '.tsx', '.jsx', '.json', '.yml', '.yaml'}:
                continue

            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue

            for pattern in patterns:
                matched = re.search(pattern, content, re.IGNORECASE)
                if not matched:
                    continue
                matches.append(
                    {
                        'file': str(file_path.relative_to(repo_path)),
                        'line': content.count('\n', 0, matched.start()) + 1,
                        'pattern': pattern,
                    }
                )
                break

        return {
            'present': bool(matches),
            'matches': matches[:20],
        }
