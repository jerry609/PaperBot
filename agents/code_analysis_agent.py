# securipaperbot/agents/code_analysis_agent.py

from typing import Dict, List, Any, Optional
import asyncio
import git
from pathlib import Path
import tempfile
from .base_agent import BaseAgent
from utils.analyzer import CodeAnalyzer


class CodeAnalysisAgent(BaseAgent):
    """负责代码仓库分析的代理"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.analyzer = CodeAnalyzer(config)
        self.temp_dir = Path(tempfile.gettempdir()) / "securipaperbot"
        self.temp_dir.mkdir(exist_ok=True)

    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """处理 GitHub 仓库分析任务"""
        repo_links = []
        
        # 适配 repo_url 参数
        if "repo_url" in kwargs:
             repo_links = [kwargs["repo_url"]]
        elif "github_links" in kwargs:
             repo_links = kwargs["github_links"]
        elif args and isinstance(args[0], list):
             repo_links = args[0]
        else:
             # 尝试从位置参数获取
             if args and isinstance(args[0], str):
                 repo_links = [args[0]]
             else:
                 raise ValueError("Invalid arguments for CodeAnalysisAgent.process")

        result = await self._process_batch(repo_links)

        # 如果是单仓库模式，尝试返回扁平化结果适配 Coordinator
        if "repo_url" in kwargs and result['analysis_results']:
            repo_result = result['analysis_results'][0]
            return self._flatten_result(repo_result)
            
        return result

    def _flatten_result(self, repo_result: Dict[str, Any]) -> Dict[str, Any]:
        """将嵌套的分析结果扁平化"""
        analysis = repo_result.get('analysis', {})
        structure = analysis.get('structure_analysis', {})
        quality = analysis.get('quality_analysis', {})
        
        # 尝试从 git 信息中获取更新时间（如果 CodeAnalyzer 支持）
        # 目前 CodeAnalyzer 似乎没有返回 updated_at
        
        flat = {
            "repo_name": repo_result.get('repo_url', '').split('/')[-1],
            "repo_url": repo_result.get('repo_url'),
            "stars": 0, # 静态分析无法获取，需 GitHub API
            "forks": 0,
            "language": structure.get('primary_language', 'Unknown'),
            "updated_at": None,
            "has_readme": structure.get('documentation', {}).get('has_readme', False),
            "reproducibility_score": quality.get('overall_score', 0) * 100,
            "quality_notes": str(quality.get('recommendations', []))
        }
        return flat

    async def _process_batch(self, github_links: List[str]) -> Dict[str, Any]:
        """处理一组GitHub仓库的分析任务（原 process 逻辑）"""
        try:
            analysis_results = []
            for link in github_links:
                result = await self._analyze_repository(link)
                if result:
                    analysis_results.append(result)

            return {
                'repositories_analyzed': len(analysis_results),
                'analysis_results': analysis_results
            }
        except Exception as e:
            self.log_error(e)
            raise

    async def _analyze_repository(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """分析单个GitHub仓库"""
        try:
            # 克隆仓库
            repo_path = await self._clone_repository(repo_url)
            if not repo_path:
                return None

            # 进行代码分析
            analysis_result = await self._perform_analysis(repo_path)

            # 清理临时文件
            await self._cleanup(repo_path)

            return {
                'repo_url': repo_url,
                'analysis': analysis_result
            }
        except Exception as e:
            self.log_error(e, {'repo_url': repo_url})
            return None

    async def _clone_repository(self, repo_url: str) -> Optional[Path]:
        """克隆GitHub仓库"""
        try:
            repo_name = repo_url.split('/')[-1]
            repo_path = self.temp_dir / repo_name

            if repo_path.exists():
                await self._cleanup(repo_path)

            git.Repo.clone_from(repo_url, str(repo_path))
            return repo_path
        except Exception as e:
            self.log_error(e, {'repo_url': repo_url})
            return None

    async def _perform_analysis(self, repo_path: Path) -> Dict[str, Any]:
        """执行代码分析"""
        analysis_tasks = [
            self._analyze_structure(repo_path),
            self._analyze_security(repo_path),
            self._analyze_quality(repo_path),
            self._analyze_dependencies(repo_path)
        ]

        results = await asyncio.gather(*analysis_tasks)

        return {
            'structure_analysis': results[0],
            'security_analysis': results[1],
            'quality_analysis': results[2],
            'dependency_analysis': results[3]
        }

    async def _analyze_structure(self, repo_path: Path) -> Dict[str, Any]:
        """分析代码结构"""
        return await self.analyzer.analyze_structure(repo_path)

    async def _analyze_security(self, repo_path: Path) -> Dict[str, Any]:
        """分析安全性"""
        return await self.analyzer.analyze_security(repo_path)

    async def _analyze_quality(self, repo_path: Path) -> Dict[str, Any]:
        """分析代码质量"""
        return await self.analyzer.analyze_quality(repo_path)

    async def _analyze_dependencies(self, repo_path: Path) -> Dict[str, Any]:
        """分析依赖关系"""
        return await self.analyzer.analyze_dependencies(repo_path)

    async def explain_code_security(self, code_snippet: str) -> str:
        """使用Claude解释代码的安全隐患"""
        prompt = f"Analyze the following code snippet for security vulnerabilities and explain them briefly:\n\n{code_snippet}"
        return await self.ask_claude(prompt, system="You are a security expert.")

    async def _cleanup(self, repo_path: Path):
        """清理临时文件"""
        try:
            import shutil
            if repo_path.exists():
                shutil.rmtree(str(repo_path))
        except Exception as e:
            self.log_error(e, {'repo_path': str(repo_path)})

    def validate_config(self) -> bool:
        """验证配置"""
        required_keys = ['github_token', 'analysis_depth', 'security_checks']
        return all(key in self.config for key in required_keys)