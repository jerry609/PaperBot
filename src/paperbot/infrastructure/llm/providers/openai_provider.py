# src/paperbot/infrastructure/llm/providers/openai_provider.py
"""
OpenAI / DeepSeek / OpenRouter Provider

支持所有 OpenAI API 兼容的服务。
"""

from __future__ import annotations

import os
import logging
import re
from typing import Any, Dict, Generator, List, Optional

from .base import LLMProvider, ProviderInfo

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("openai 库未安装: pip install openai")


class OpenAIProvider(LLMProvider):
    """
    OpenAI 兼容的 LLM Provider
    
    支持:
    - OpenAI (gpt-4o, gpt-4o-mini)
    - DeepSeek (deepseek-chat, deepseek-coder)
    - OpenRouter (多模型路由)
    """
    
    ALLOWED_PARAMS = {
        "temperature", "top_p", "presence_penalty", 
        "frequency_penalty", "max_tokens", "timeout"
    }
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        cost_tier: int = 1,
    ):
        """
        初始化 OpenAI Provider
        
        Args:
            api_key: API 密钥
            model_name: 模型名称
            base_url: 自定义 API 地址
            timeout: 请求超时 (秒)
            cost_tier: 成本层级 (1-3)
        """
        if OpenAI is None:
            raise RuntimeError("需要安装 openai 库: pip install openai")
        
        if not api_key:
            raise ValueError("API key 不能为空")
        
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.cost_tier = cost_tier
        
        # 解析超时
        if timeout is not None:
            self.timeout = timeout
        else:
            timeout_env = os.getenv("LLM_REQUEST_TIMEOUT", "1800")
            try:
                self.timeout = float(timeout_env)
            except ValueError:
                self.timeout = 1800.0
        
        # 检测提供商
        self._provider_name = self._detect_provider()
        
        # 初始化客户端
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        logger.info(f"OpenAIProvider 初始化: {self}")
    
    def _detect_provider(self) -> str:
        """检测实际提供商"""
        if self.base_url:
            url_lower = self.base_url.lower()
            if "deepseek" in url_lower:
                return "deepseek"
            elif "anthropic" in url_lower or "claude" in url_lower:
                return "anthropic-proxy"
            elif "openrouter" in url_lower:
                return "openrouter"
        
        model_lower = self.model_name.lower()
        if "gpt" in model_lower:
            return "openai"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "claude" in model_lower:
            return "anthropic-proxy"
        
        return "openai-compatible"
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """非流式调用"""
        extra_params = {
            k: v for k, v in kwargs.items()
            if k in self.ALLOWED_PARAMS and v is not None
        }

        timeout = extra_params.pop("timeout", self.timeout)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=timeout,
            **extra_params,
        )

        if response.choices and response.choices[0].message:
            msg = response.choices[0].message
            content = msg.content or ""
            # Some thinking models (GLM4.7) put the answer in reasoning_content
            # when content is empty
            if not content.strip():
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    content = str(rc)
            content = self._strip_thinking_tags(content)
            return content.strip() if content else ""
        return ""
    
    def stream_invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """流式调用"""
        extra_params = {
            k: v for k, v in kwargs.items()
            if k in self.ALLOWED_PARAMS and v is not None
        }
        extra_params["stream"] = True

        timeout = extra_params.pop("timeout", self.timeout)

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=timeout,
                **extra_params,
            )

            in_think = False
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    text = delta.content or ""
                    # Skip reasoning_content chunks from thinking models
                    if not text:
                        rc = getattr(delta, "reasoning_content", None)
                        if rc:
                            continue
                    # Filter out <think>...</think> inline tags from MiniMax
                    if "<think>" in text:
                        in_think = True
                    if in_think:
                        if "</think>" in text:
                            in_think = False
                            text = text.split("</think>", 1)[1]
                        else:
                            continue
                    if text:
                        yield text
        except Exception as e:
            logger.error(f"流式请求失败: {e}")
            raise
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name=self._provider_name,
            model_name=self.model_name,
            api_base=self.base_url or "https://api.openai.com",
            cost_tier=self.cost_tier,
            max_tokens=4096,
            supports_streaming=True,
        )

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove <think>...</think> blocks from thinking model output (e.g. MiniMax M2.1)."""
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
