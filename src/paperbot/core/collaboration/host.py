# src/paperbot/core/collaboration/host.py
"""
主持人模块：参考 BettaFish ForumHost，生成多 Agent 协作引导语。
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .messages import AgentMessage

logger = logging.getLogger(__name__)


HOST_SYSTEM_PROMPT = """你是多 Agent 协作的主持人，负责：
1) 事件梳理：提炼关键发现和冲突
2) 观点整合：指出共识与分歧，标注来源 Agent
3) 纠错提醒：发现逻辑/事实漏洞要直说
4) 趋势与风险：提示潜在风险或缺口
5) 下一步行动：给出 2-3 条可执行的检索/验证/改写建议
输出务必简洁、有条理，1000字以内，保持客观。"""


@dataclass
class HostConfig:
    enabled: bool = False
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.9


class HostOrchestrator:
    """主持人 orchestrator，按需生成引导语。"""

    def __init__(self, config: HostConfig):
        self.config = config
        self.llm = None
        
        if config.enabled and config.api_key and config.model:
            try:
                # 延迟导入以避免循环依赖
                from src.paperbot.infrastructure.llm import LLMClient
                self.llm = LLMClient(
                    api_key=config.api_key,
                    model_name=config.model,
                    base_url=config.base_url,
                )
            except ImportError:
                try:
                    from paperbot.core.llm_client import LLMClient
                    self.llm = LLMClient(
                        api_key=config.api_key,
                        model_name=config.model,
                        base_url=config.base_url,
                    )
                except Exception as exc:
                    logger.warning(f"Host LLM 初始化失败，自动降级为禁用模式: {exc}")
                    self.llm = None
                    self.config.enabled = False
            except Exception as exc:
                logger.warning(f"Host LLM 初始化失败，自动降级为禁用模式: {exc}")
                self.llm = None
                self.config.enabled = False
        else:
            logger.info("Host 未启用或缺少密钥/模型，跳过主持人功能")

    def is_available(self) -> bool:
        return self.config.enabled and self.llm is not None

    def build_user_prompt(self, messages: List[AgentMessage], context: Dict[str, Any]) -> str:
        """
        将消息和上下文压缩为主持人输入。
        """
        lines = []
        if context:
            lines.append("【任务上下文】")
            for k, v in context.items():
                lines.append(f"- {k}: {v}")
        if messages:
            lines.append("\n【最新发言】(按时间顺序)")
            for msg in messages:
                content = msg.content
                if isinstance(content, dict):
                    content = str(content)
                content = str(content).replace("\n", " ")
                lines.append(
                    f"[{msg.round_index}][{msg.stage or 'stage'}][{msg.sender}] {content[:800]}"
                )
        return "\n".join(lines)

    def generate_guidance(
        self,
        messages: List[AgentMessage],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        生成主持人引导语。失败时返回 None。
        """
        if not self.is_available():
            return None
        try:
            user_prompt = self.build_user_prompt(messages, context or {})
            return self.llm.invoke(
                HOST_SYSTEM_PROMPT,
                user_prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        except Exception as exc:
            logger.warning(f"生成主持人发言失败: {exc}")
            return None

