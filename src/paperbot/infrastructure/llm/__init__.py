"""
LLM 客户端抽象层。

P4 增强:
- LLMProvider 抽象基类
- 多后端支持 (OpenAI, Anthropic, Ollama)
- ModelRouter 成本路由
"""

from .base import LLMClient, LLMClientProtocol
from .providers.base import LLMProvider, ProviderInfo
from .router import ModelRouter, ModelConfig, RouterConfig, TaskType

__all__ = [
    # 兼容层
    "LLMClient", 
    "LLMClientProtocol",
    # P4 新增
    "LLMProvider",
    "ProviderInfo",
    "ModelRouter",
    "ModelConfig",
    "RouterConfig",
    "TaskType",
]

