# paperbot/core/llm_client.py
"""
LLM 客户端兼容层

从 infrastructure 层导出 LLMClient，保持向后兼容。
"""

from paperbot.infrastructure.llm.base import LLMClient, LLMClientProtocol

__all__ = ["LLMClient", "LLMClientProtocol"]

