# src/paperbot/infrastructure/llm/providers/__init__.py
"""
LLM Provider 提供商抽象层

P4 增强:
- 统一的 LLMProvider 接口
- 多后端支持 (OpenAI, Anthropic, Ollama)
"""

from .base import LLMProvider, ProviderInfo
from .openai_provider import OpenAIProvider

__all__ = [
    "LLMProvider",
    "ProviderInfo", 
    "OpenAIProvider",
]

# 可选导入 (依赖可能未安装)
try:
    from .anthropic_provider import AnthropicProvider
    __all__.append("AnthropicProvider")
except ImportError:
    pass

try:
    from .ollama_provider import OllamaProvider
    __all__.append("OllamaProvider")
except ImportError:
    pass
