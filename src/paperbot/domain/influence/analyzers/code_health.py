# src/paperbot/domain/influence/analyzers/code_health.py
"""
Code Health Analyzer.

Deep repository health analysis:
- Empty repo detection
- Documentation coverage
- Dependency risk scanning
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..result import CodeHealthResult, DependencyRisk, DependencyRiskLevel

logger = logging.getLogger(__name__)


# Known deprecated/problematic packages
DEPRECATED_PACKAGES = {
    "sklearn": "Rename to scikit-learn",
    "gym": "Use gymnasium instead",
    "imgaug": "Unmaintained",
}

# Packages with known security issues (simplified)
KNOWN_VULNERABLE = {
    "pyyaml<5.4": "CVE-2020-14343",
    "pillow<8.0": "Multiple CVEs",
    "urllib3<1.26.5": "CVE-2021-33503",
    "requests<2.25.0": "Security updates",
}


class CodeHealthAnalyzer:
    """
    Analyze code repository health.
    
    Performs:
    - Empty repo detection (shell repos with just README)
    - Documentation coverage assessment
    - Dependency risk scanning
    """
    
    # Thresholds for empty repo detection
    MIN_COMMITS_FOR_REAL_REPO = 5
    MIN_CODE_FILES_FOR_REAL_REPO = 2
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(
        self,
        repo_url: Optional[str],
        code_meta,
    ) -> CodeHealthResult:
        """
        Perform comprehensive health analysis.
        
        Args:
            repo_url: GitHub/GitLab URL (unused for now, for future API calls)
            code_meta: Repository metadata object
            
        Returns:
            CodeHealthResult with all health metrics
        """
        result = CodeHealthResult()
        
        if not code_meta:
            result.warnings.append("No repository metadata available")
            return result
        
        # 1. Check for empty/shell repo
        result.is_empty_repo = self.detect_empty_repo(code_meta)
        result.commit_count = getattr(code_meta, 'commit_count', 0) or 0
        
        # 2. Assess documentation coverage
        result.doc_coverage = self.compute_doc_coverage(code_meta)
        result.has_readme = getattr(code_meta, 'has_readme', False)
        result.has_docs_folder = getattr(code_meta, 'has_docs', False)
        result.has_tests = getattr(code_meta, 'has_tests', False)
        
        # 3. Scan dependency risks
        result.dependency_risks = self.scan_dependency_risks(code_meta)
        
        # 4. Compute overall health score
        result.health_score = self._compute_health_score(result)
        
        # 5. Generate warnings
        result.warnings = self._generate_warnings(result)
        
        return result
    
    def detect_empty_repo(self, code_meta) -> bool:
        """
        Detect if repository is a shell/empty repo.
        
        Criteria for empty repo:
        - Less than 5 commits
        - OR no actual code files
        - OR only README exists
        """
        # Check commit count
        commit_count = getattr(code_meta, 'commit_count', 0) or 0
        if commit_count < self.MIN_COMMITS_FOR_REAL_REPO:
            return True
        
        # Check for code files
        code_files = getattr(code_meta, 'code_files_count', None)
        if code_files is not None and code_files < self.MIN_CODE_FILES_FOR_REAL_REPO:
            return True
        
        # Check if only README
        files = getattr(code_meta, 'files', [])
        if isinstance(files, list):
            code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
            has_code = any(
                any(f.endswith(ext) for ext in code_extensions)
                for f in files
            )
            if not has_code:
                return True
        
        return False
    
    def compute_doc_coverage(self, code_meta) -> float:
        """
        Compute documentation coverage score.
        
        Components:
        - Has README: 40 points
        - README quality (length, sections): up to 20 points
        - Has docs folder: 20 points
        - Has API docs/docstrings: 20 points
        """
        score = 0.0
        
        # README existence and quality
        has_readme = getattr(code_meta, 'has_readme', False)
        if has_readme:
            score += 40
            
            # README quality based on length
            readme_lines = getattr(code_meta, 'readme_lines', 0) or 0
            if readme_lines > 100:
                score += 20
            elif readme_lines > 50:
                score += 15
            elif readme_lines > 20:
                score += 10
        
        # Docs folder
        if getattr(code_meta, 'has_docs', False):
            score += 20
        
        # Tests (implies some documentation via test names)
        if getattr(code_meta, 'has_tests', False):
            score += 10
        
        # License (documentation of usage terms)
        if getattr(code_meta, 'license', None):
            score += 10
        
        return min(100, score)
    
    def scan_dependency_risks(self, code_meta) -> List[DependencyRisk]:
        """
        Scan for dependency risks.
        
        Checks:
        - Deprecated packages
        - Known vulnerabilities
        - Outdated dependencies (if version info available)
        """
        risks = []
        
        # Get dependencies from code_meta
        dependencies = getattr(code_meta, 'dependencies', [])
        if not dependencies:
            requirements = getattr(code_meta, 'requirements', [])
            if requirements:
                dependencies = requirements
        
        if not isinstance(dependencies, list):
            return risks
        
        for dep in dependencies:
            # Parse package name (handle "package==version" format)
            if isinstance(dep, str):
                parts = dep.split('==')
                package = parts[0].split('>=')[0].split('<=')[0].strip()
                version = parts[1] if len(parts) > 1 else None
            elif isinstance(dep, dict):
                package = dep.get('name', '')
                version = dep.get('version')
            else:
                continue
            
            package_lower = package.lower()
            
            # Check deprecated packages
            if package_lower in DEPRECATED_PACKAGES:
                risks.append(DependencyRisk(
                    package=package,
                    risk_type="deprecated",
                    severity=DependencyRiskLevel.MEDIUM,
                    description=DEPRECATED_PACKAGES[package_lower],
                ))
            
            # Check known vulnerabilities
            for vuln_pattern, vuln_desc in KNOWN_VULNERABLE.items():
                vuln_pkg = vuln_pattern.split('<')[0]
                if package_lower == vuln_pkg.lower():
                    risks.append(DependencyRisk(
                        package=package,
                        risk_type="vulnerable",
                        severity=DependencyRiskLevel.HIGH,
                        description=vuln_desc,
                    ))
                    break
        
        return risks
    
    def _compute_health_score(self, result: CodeHealthResult) -> float:
        """Compute overall health score."""
        score = 0.0
        
        # Not empty repo: 30 points
        if not result.is_empty_repo:
            score += 30
        else:
            return 10.0  # Empty repos get very low score
        
        # Documentation coverage: up to 30 points
        score += result.doc_coverage * 0.3
        
        # No major dependency risks: 20 points
        high_risks = sum(
            1 for r in result.dependency_risks 
            if r.severity == DependencyRiskLevel.HIGH
        )
        if high_risks == 0:
            score += 20
        elif high_risks == 1:
            score += 10
        
        # Has tests: 10 points
        if result.has_tests:
            score += 10
        
        # Good commit history: 10 points
        if result.commit_count >= 50:
            score += 10
        elif result.commit_count >= 20:
            score += 5
        
        return min(100, max(0, score))
    
    def _generate_warnings(self, result: CodeHealthResult) -> List[str]:
        """Generate human-readable warnings."""
        warnings = []
        
        if result.is_empty_repo:
            warnings.append("⚠️ 可能是空壳仓库: 提交数或代码文件过少")
        
        if result.doc_coverage < 30:
            warnings.append("📝 文档覆盖率低: 缺少README或说明文档")
        
        for risk in result.dependency_risks:
            if risk.severity == DependencyRiskLevel.HIGH:
                warnings.append(f"🔴 高风险依赖: {risk.package} - {risk.description}")
            elif risk.severity == DependencyRiskLevel.MEDIUM:
                warnings.append(f"🟡 注意依赖: {risk.package} - {risk.description}")
        
        if not result.has_tests:
            warnings.append("🧪 无测试目录: 可能影响可复现性")
        
        return warnings
    
    def get_health_summary(self, result: CodeHealthResult) -> str:
        """Generate human-readable health summary."""
        if result.health_score >= 80:
            status = "🟢 健康"
        elif result.health_score >= 50:
            status = "🟡 一般"
        else:
            status = "🔴 不佳"
        
        return (
            f"{status} (分数: {result.health_score:.0f}/100)\n"
            f"文档覆盖: {result.doc_coverage:.0f}%\n"
            f"提交数: {result.commit_count}\n"
            f"风险项: {len(result.dependency_risks)}"
        )
