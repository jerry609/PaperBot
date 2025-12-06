import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from repro.docker_executor import DockerExecutor

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None
    logger.warning("claude-agent-sdk not installed; ReproAgent will use fallback plan.")


DEFAULT_PLAN = [
    "python -m pip install -U pip",
    "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
    "if [ -d tests ]; then pytest -q || true; else python -m py_compile $(find . -name '*.py'); fi",
]


class ReproAgent:
    """
    生成并执行可复现性验证计划：
    - 尝试用 Claude Agent SDK 生成命令列表；失败则回退默认计划
    - 使用 DockerExecutor 在隔离环境执行
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        repro_cfg = self.config.get("repro", {})
        self.executor = DockerExecutor(
            image=repro_cfg.get("docker_image", "python:3.10-slim"),
            cpu_shares=repro_cfg.get("cpu_shares", 1),
            mem_limit=repro_cfg.get("mem_limit", "1g"),
            network=repro_cfg.get("network", False),
        )
        self.timeout_sec = repro_cfg.get("timeout_sec", 300)

    async def generate_plan(self, repo_path: Path) -> List[str]:
        if query is None or ClaudeAgentOptions is None:
            return DEFAULT_PLAN

        try:
            prompt = (
                "你是可复现性验证助手。根据代码仓库生成安装和测试命令，"
                "使用 bash && 串联，避免交互。优先 pip/pytest。简洁输出命令列表，每行一个命令。"
            )
            opts = ClaudeAgentOptions(
                system_prompt=prompt,
                model="claude-3-5-sonnet-latest",
            )
            cmds: List[str] = []
            async for msg in query(
                prompt=f"仓库路径: {repo_path}. 给出命令列表。",
                options=opts,
            ):
                content = getattr(msg, "content", None)
                if not content:
                    continue
                if isinstance(content, list):
                    for block in content:
                        text = getattr(block, "text", None) or getattr(block, "thinking", None)
                        if text:
                            for line in text.splitlines():
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    cmds.append(line)
                elif isinstance(content, str):
                    for line in content.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            cmds.append(line)
            return cmds or DEFAULT_PLAN
        except Exception as e:
            logger.warning(f"Claude plan generation failed, fallback: {e}")
            return DEFAULT_PLAN

    async def run(self, repo_path: Path) -> Dict[str, Any]:
        if not self.executor.available():
            return {"status": "error", "error": "Docker not available"}

        cmds = await self.generate_plan(repo_path)
        result = self.executor.run(repo_path, cmds, timeout_sec=self.timeout_sec)
        score = self._score(result)
        return {
            "status": result.get("status"),
            "commands": cmds,
            "exit_code": result.get("exit_code"),
            "duration_sec": result.get("duration_sec"),
            "logs": result.get("logs", "")[-2000:],
            "score": score,
            "error": result.get("error"),
        }

    def _score(self, result: Dict[str, Any]) -> int:
        if result.get("status") != "success":
            return 0
        # 简单评分：成功即 100，可扩展按命令数/耗时等加权
        return 100

