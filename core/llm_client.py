"""
统一的 OpenAI 兼容 LLM 客户端封装

来源: BettaFish/QueryEngine/llms/base.py
适配: PaperBot 学者追踪系统

提供:
- 统一的非流式/流式调用
- 可选重试机制
- 字节安全拼接（避免UTF-8多字节截断）
- 模型元信息查询
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Generator, List
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("openai 库未安装，LLM 功能将不可用")

try:
    from utils.retry_helper import with_retry, LLM_RETRY_CONFIG
except ImportError:
    # 如果 retry_helper 不可用，提供空装饰器
    def with_retry(config=None):
        def decorator(func):
            return func
        return decorator
    LLM_RETRY_CONFIG = None


class LLMClient:
    """
    OpenAI 兼容的 LLM 客户端封装
    
    支持 OpenAI、DeepSeek、Claude (via OpenAI proxy) 等 API
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: API 密钥
            model_name: 模型名称 (如 gpt-4, deepseek-chat, claude-3-opus)
            base_url: 自定义 API 地址，默认为 OpenAI 官方
            timeout: 请求超时时间（秒）
        """
        if OpenAI is None:
            raise RuntimeError("需要安装 openai 库: pip install openai")
        
        if not api_key:
            raise ValueError("LLM API key 不能为空")
        if not model_name:
            raise ValueError("模型名称不能为空")

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.provider = self._detect_provider(model_name, base_url)
        
        # 解析超时配置
        if timeout is not None:
            self.timeout = timeout
        else:
            timeout_env = os.getenv("LLM_REQUEST_TIMEOUT", "1800")
            try:
                self.timeout = float(timeout_env)
            except ValueError:
                self.timeout = 1800.0

        # 初始化 OpenAI 客户端
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,  # 由我们的 retry_helper 处理重试
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        logger.info(f"LLM 客户端已初始化: {self.get_model_info()}")

    def _detect_provider(self, model_name: str, base_url: Optional[str]) -> str:
        """根据模型名称和 base_url 检测提供商"""
        if base_url:
            if "deepseek" in base_url.lower():
                return "deepseek"
            elif "anthropic" in base_url.lower() or "claude" in base_url.lower():
                return "anthropic"
            elif "openrouter" in base_url.lower():
                return "openrouter"
        
        if "gpt" in model_name.lower():
            return "openai"
        elif "deepseek" in model_name.lower():
            return "deepseek"
        elif "claude" in model_name.lower():
            return "anthropic"
        
        return "unknown"

    @with_retry(LLM_RETRY_CONFIG)
    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        include_time: bool = False,
        **kwargs
    ) -> str:
        """
        非流式调用 LLM
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            include_time: 是否在 prompt 中包含当前时间
            **kwargs: 额外参数 (temperature, top_p 等)
            
        Returns:
            LLM 响应文本
        """
        if include_time:
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            user_prompt = f"当前时间: {current_time}\n\n{user_prompt}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        allowed_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"}
        extra_params = {
            key: value 
            for key, value in kwargs.items() 
            if key in allowed_keys and value is not None
        }

        timeout = kwargs.pop("timeout", self.timeout)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=timeout,
            **extra_params,
        )

        if response.choices and response.choices[0].message:
            return self._validate_response(response.choices[0].message.content)
        return ""

    def invoke_with_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        使用自定义消息列表调用 LLM
        
        Args:
            messages: 消息列表 [{"role": "system", "content": "..."}, ...]
            **kwargs: 额外参数
            
        Returns:
            LLM 响应文本
        """
        allowed_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"}
        extra_params = {
            key: value 
            for key, value in kwargs.items() 
            if key in allowed_keys and value is not None
        }

        timeout = kwargs.pop("timeout", self.timeout)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=timeout,
            **extra_params,
        )

        if response.choices and response.choices[0].message:
            return self._validate_response(response.choices[0].message.content)
        return ""

    def stream_invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        include_time: bool = False,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式调用 LLM，逐步返回响应内容
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            include_time: 是否包含当前时间
            **kwargs: 额外参数
            
        Yields:
            响应文本块
        """
        if include_time:
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            user_prompt = f"当前时间: {current_time}\n\n{user_prompt}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        allowed_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"}
        extra_params = {
            key: value 
            for key, value in kwargs.items() 
            if key in allowed_keys and value is not None
        }
        extra_params["stream"] = True

        timeout = kwargs.pop("timeout", self.timeout)

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=timeout,
                **extra_params,
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"流式请求失败: {str(e)}")
            raise

    @with_retry(LLM_RETRY_CONFIG)
    def stream_invoke_to_string(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        流式调用并安全拼接为完整字符串（避免 UTF-8 多字节字符截断）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 额外参数
            
        Returns:
            完整的响应字符串
        """
        byte_chunks = []
        for chunk in self.stream_invoke(system_prompt, user_prompt, **kwargs):
            byte_chunks.append(chunk.encode('utf-8'))
        
        if byte_chunks:
            return b''.join(byte_chunks).decode('utf-8', errors='replace')
        return ""

    @staticmethod
    def _validate_response(response: Optional[str]) -> str:
        """验证并清理响应"""
        if response is None:
            return ""
        return response.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "api_base": self.base_url or "https://api.openai.com",
            "timeout": self.timeout,
        }

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model_name}, provider={self.provider})"
