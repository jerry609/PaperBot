# securipaperbot/agents/base_agent.py

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
import os
from anthropic import AsyncAnthropic

class BaseAgent(ABC):
    """基础代理类，定义所有代理的通用接口"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Anthropic client
        api_key = self.config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None
        
        if not self.client:
            self.logger.warning("Anthropic API key not found. AI features will be disabled.")

    @abstractmethod
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """处理方法，需要被子类实现"""
        pass

    async def ask_claude(self, prompt: str, system: str = "", max_tokens: int = 1000) -> str:
        """使用Claude模型生成回答"""
        if not self.client:
            self.logger.warning("Attempted to use Claude without API key.")
            return "AI capabilities disabled: No API key provided."

        try:
            model = self.config.get('apis', {}).get('anthropic', {}).get('model', 'claude-3-5-sonnet-20241022')
            
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                temperature=self.config.get('apis', {}).get('anthropic', {}).get('temperature', 0.0)
            )
            return response.content[0].text
            
        except Exception as e:
            self.log_error(e, {"context": "ask_claude"})
            return f"Error communicating with Claude: {str(e)}"

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        return True

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """记录错误信息"""
        self.logger.error(f"Error in {self.__class__.__name__}: {str(error)}",
                          extra={"context": context})

    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录信息"""
        self.logger.info(message, extra={"context": context})