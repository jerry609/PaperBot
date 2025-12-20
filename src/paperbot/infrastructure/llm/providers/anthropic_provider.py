# src/paperbot/infrastructure/llm/providers/anthropic_provider.py
"""
Anthropic Claude Provider (Native SDK)

使用原生 Anthropic SDK 与 Claude 交互。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional

from .base import LLMProvider, ProviderInfo

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None
    logger.warning("anthropic 库未安装: pip install anthropic")


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude Native Provider
    
    支持:
    - claude-3-5-sonnet-20241022
    - claude-3-opus-20240229
    - claude-3-haiku-20240307
    """
    
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    
    # Claude 模型成本层级
    MODEL_COST_TIERS = {
        "claude-3-5-sonnet": 2,
        "claude-3-opus": 3,
        "claude-3-haiku": 1,
    }
    
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        cost_tier: Optional[int] = None,
    ):
        """
        初始化 Anthropic Provider
        
        Args:
            api_key: Anthropic API 密钥
            model_name: 模型名称
            max_tokens: 最大输出 tokens
            cost_tier: 成本层级 (自动检测如果未指定)
        """
        if anthropic is None:
            raise RuntimeError("需要安装 anthropic 库: pip install anthropic")
        
        if not api_key:
            raise ValueError("API key 不能为空")
        
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # 自动检测成本层级
        if cost_tier is not None:
            self.cost_tier = cost_tier
        else:
            self.cost_tier = self._detect_cost_tier()
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"AnthropicProvider 初始化: {self}")
    
    def _detect_cost_tier(self) -> int:
        """根据模型名称检测成本层级"""
        model_lower = self.model_name.lower()
        for prefix, tier in self.MODEL_COST_TIERS.items():
            if prefix in model_lower:
                return tier
        return 2  # 默认中等
    
    def _convert_messages(
        self, 
        messages: List[Dict[str, str]]
    ) -> tuple[str, List[Dict[str, str]]]:
        """
        转换 OpenAI 格式消息到 Anthropic 格式
        
        Anthropic API 要求 system prompt 单独传递
        """
        system_prompt = ""
        converted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            else:
                # Anthropic 使用 "assistant" 而非其他角色
                anthropic_role = "assistant" if role == "assistant" else "user"
                converted.append({"role": anthropic_role, "content": content})
        
        return system_prompt, converted
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """非流式调用"""
        system_prompt, converted_messages = self._convert_messages(messages)
        
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", 1.0)
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
            messages=converted_messages,
            temperature=temperature,
        )
        
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return ""
    
    def stream_invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """流式调用"""
        system_prompt, converted_messages = self._convert_messages(messages)
        
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", 1.0)
        
        try:
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
                messages=converted_messages,
                temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic 流式请求失败: {e}")
            raise
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name="anthropic",
            model_name=self.model_name,
            api_base="https://api.anthropic.com",
            cost_tier=self.cost_tier,
            max_tokens=self.max_tokens,
            supports_streaming=True,
        )
