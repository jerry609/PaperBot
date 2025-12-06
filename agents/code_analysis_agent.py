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
        elif args and isinstance(args[0], str):
             repo_links = [args[0]]
        else:
             # 无仓库时返回占位结果
             return self._placeholder(None, reason="no_repository_provided")

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
        
        meta = repo_result.get("meta", {})
        flat = {
            "repo_name": repo_result.get('repo_url', '').split('/')[-1],
            "repo_url": repo_result.get('repo_url'),
            "stars": meta.get("stars"),
            "forks": meta.get("forks"),
            "language": structure.get('primary_language', 'Unknown'),
            "updated_at": meta.get("last_commit_at"),
            "has_readme": structure.get('documentation', {}).get('has_readme', False),
            "reproducibility_score": quality.get('overall_score', 0) * 100,
            "quality_notes": str(quality.get('recommendations', [])),
            "confidence": self._compute_confidence(meta, structure),
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
                return self._placeholder(repo_url, reason="clone_failed")

            # 进行代码分析
            analysis_result = await self._perform_analysis(repo_path)
            meta = self._extract_repo_meta(repo_path, repo_url)

            # 清理临时文件
            await self._cleanup(repo_path)

            return {
                'repo_url': repo_url,
                'analysis': analysis_result,
                'meta': meta,
            }
        except Exception as e:
            self.log_error(e, {'repo_url': repo_url})
            return self._placeholder(repo_url, reason="analysis_failed")

    def _placeholder(self, repo_url: Optional[str], reason: str) -> Dict[str, Any]:
        repo_name = (repo_url or "").split("/")[-1] if repo_url else None
        return {
            "repo_url": repo_url,
            "analysis": {},
            "reason": reason,
            "placeholder": True,
            "repo_name": repo_name,
            "stars": None,
            "forks": None,
            "language": None,
            "updated_at": None,
            "has_readme": False,
            "reproducibility_score": None,
            "quality_notes": f"Repository unavailable: {reason}",
        }

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

    def _compute_confidence(self, meta: Dict[str, Any], structure: Dict[str, Any]) -> float:
        """简单信心分：有 README + 近期提交 + stars/forks 加权"""
        score = 0.3  # base
        if structure.get('documentation', {}).get('has_readme'):
            score += 0.2
        try:
            from datetime import datetime, timedelta
            last = meta.get("last_commit_at")
            if last:
                dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                if datetime.now(dt.tzinfo) - dt <= timedelta(days=180):
                    score += 0.2
        except Exception:
            pass
        stars = meta.get("stars") or 0
        forks = meta.get("forks") or 0
        if stars or forks:
            score += 0.2
        return round(min(1.0, score), 2)
    def _extract_repo_meta(self, repo_path: Path, repo_url: str) -> Dict[str, Any]:
        """获取静态元信息：last commit 时间，GitHub stars/forks（若有 token）"""
        meta: Dict[str, Any] = {}
        try:
            repo = git.Repo(repo_path)
            last_commit = next(repo.iter_commits(max_count=1), None)
            if last_commit:
                meta["last_commit_at"] = last_commit.committed_datetime.isoformat()
        except Exception as e:
            self.log_error(e, {"repo_meta": "commit_time"})

        try:
            token = self.config.get("api", {}).get("github_token") or self.config.get("github_token")
            if token and "github.com" in repo_url:
                import requests
                parts = repo_url.rstrip("/").split("/")
                if len(parts) >= 2:
                    owner, name = parts[-2], parts[-1]
                    resp = requests.get(
                        f"https://api.github.com/repos/{owner}/{name}",
                        headers={"Authorization": f"token {token}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        meta["stars"] = data.get("stargazers_count")
                        meta["forks"] = data.get("forks_count")
        except Exception as e:
            self.log_error(e, {"repo_meta": "github_api"})

        return meta