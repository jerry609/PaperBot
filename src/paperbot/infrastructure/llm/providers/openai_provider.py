# src/paperbot/infrastructure/llm/providers/openai_provider.py
"""
OpenAI / DeepSeek / OpenRouter Provider

支持所有 OpenAI API 兼容的服务。
"""

from __future__ import annotations

import os
import logging
import re
import time
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
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_tokens",
        "timeout",
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
        self.max_transient_retries = max(0, int(os.getenv("LLM_TRANSIENT_RETRIES", "2") or 0))
        try:
            self.retry_backoff_sec = max(0.0, float(os.getenv("LLM_RETRY_BACKOFF_SEC", "2") or 0.0))
        except ValueError:
            self.retry_backoff_sec = 2.0

        # 初始化客户端
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        logger.info(f"OpenAIProvider 初始化: {self}")

    def _create_completion_with_fallback(
        self,
        *,
        messages: List[Dict[str, str]],
        timeout: float,
        extra_params: Dict[str, Any],
        stream: bool,
    ):
        request_kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            timeout=timeout,
            **extra_params,
        )
        if stream:
            request_kwargs["stream"] = True

        attempt = 0
        system_fallback_applied = False
        current_kwargs = dict(request_kwargs)

        while True:
            try:
                return self.client.chat.completions.create(**current_kwargs)
            except Exception as exc:
                current_messages = current_kwargs.get("messages") or []
                if (
                    not system_fallback_applied
                    and self._should_retry_without_system(exc, current_messages)
                ):
                    retry_kwargs = dict(current_kwargs)
                    retry_kwargs["messages"] = self._merge_system_into_user(current_messages)
                    current_kwargs = retry_kwargs
                    system_fallback_applied = True
                    continue
                if self._should_retry_transient(exc) and attempt < self.max_transient_retries:
                    delay = self.retry_backoff_sec * (2 ** attempt)
                    logger.warning(
                        "Transient LLM request failed provider=%s model=%s attempt=%s error=%s",
                        self._provider_name,
                        self.model_name,
                        attempt + 1,
                        exc,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    attempt += 1
                    continue
                raise

    @staticmethod
    def _should_retry_without_system(exc: Exception, messages: List[Dict[str, str]]) -> bool:
        if not messages or messages[0].get("role") != "system":
            return False
        text = str(exc)
        lowered = text.lower()
        markers = (
            "developer instruction is not enabled",
            "system message is not supported",
            "system role is not supported",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _should_retry_transient(exc: Exception) -> bool:
        lowered = str(exc).lower()
        markers = (
            "error code: 429",
            "rate limit",
            "rate-limit",
            "temporarily rate-limited",
            "timeout",
            "connection reset",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _merge_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return []
        system_chunks = [msg.get("content", "") for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]
        if not system_chunks:
            return [dict(msg) for msg in messages]
        merged_prefix = "\n\n".join(
            chunk.strip() for chunk in system_chunks if chunk and chunk.strip()
        )
        if not non_system:
            return [{"role": "user", "content": merged_prefix}]
        first = dict(non_system[0])
        first_content = str(first.get("content") or "").strip()
        first["content"] = (
            f"System instruction:\n{merged_prefix}\n\nUser request:\n{first_content}"
            if merged_prefix
            else first_content
        )
        return [first] + non_system[1:]

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

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """非流式调用"""
        extra_params = {
            k: v for k, v in kwargs.items() if k in self.ALLOWED_PARAMS and v is not None
        }

        timeout = extra_params.pop("timeout", self.timeout)

        response = self._create_completion_with_fallback(
            messages=messages,
            timeout=timeout,
            extra_params=extra_params,
            stream=False,
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

    def stream_invoke(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """流式调用"""
        extra_params = {
            k: v for k, v in kwargs.items() if k in self.ALLOWED_PARAMS and v is not None
        }
        extra_params["stream"] = True

        timeout = extra_params.pop("timeout", self.timeout)

        try:
            stream = self._create_completion_with_fallback(
                messages=messages,
                timeout=timeout,
                extra_params=extra_params,
                stream=True,
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
