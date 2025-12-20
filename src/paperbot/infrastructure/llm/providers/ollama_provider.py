# src/paperbot/infrastructure/llm/providers/ollama_provider.py
"""
Ollama 本地模型 Provider

支持本地运行的开源模型 (Llama, DeepSeek-Coder 等)。
"""

from __future__ import annotations

import logging
import requests
from typing import Any, Dict, Generator, List, Optional

from .base import LLMProvider, ProviderInfo

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama 本地模型 Provider
    
    支持:
    - llama3 / llama3.1
    - deepseek-coder
    - codellama
    - mistral
    
    需要本地运行 Ollama: https://ollama.ai
    """
    
    DEFAULT_MODEL = "llama3"
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
    ):
        """
        初始化 Ollama Provider
        
        Args:
            model_name: 模型名称 (需要已 pull)
            base_url: Ollama API 地址
            timeout: 请求超时 (秒)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # 验证连接
        self._check_connection()
        logger.info(f"OllamaProvider 初始化: {self}")
    
    def _check_connection(self) -> None:
        """检查 Ollama 服务是否可用"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                logger.warning(f"Ollama 服务响应异常: {resp.status_code}")
        except requests.RequestException as e:
            logger.warning(f"无法连接到 Ollama ({self.base_url}): {e}")
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        将消息列表转换为单个 prompt
        
        Ollama 的 /api/generate 端点使用单个 prompt
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """非流式调用"""
        prompt = self._build_prompt(messages)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
            }
        }
        
        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            
            data = resp.json()
            return data.get("response", "").strip()
        except requests.RequestException as e:
            logger.error(f"Ollama 请求失败: {e}")
            raise RuntimeError(f"Ollama 调用失败: {e}")
    
    def stream_invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """流式调用"""
        prompt = self._build_prompt(messages)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
            }
        }
        
        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]
        
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
        except requests.RequestException as e:
            logger.error(f"Ollama 流式请求失败: {e}")
            raise RuntimeError(f"Ollama 流式调用失败: {e}")
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            provider_name="ollama",
            model_name=self.model_name,
            api_base=self.base_url,
            cost_tier=0,  # 本地免费
            max_tokens=4096,
            supports_streaming=True,
        )
