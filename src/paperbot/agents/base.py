# src/paperbot/agents/base.py
"""
基础代理类，定义所有代理的通用接口。
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
import os
from anthropic import AsyncAnthropic

from paperbot.core.abstractions import Executable, ExecutionResult, ensure_execution_result
from paperbot.core.di import Container
from .mixins.json_parser import JSONParserMixin, JSONParseError


class BaseAgent(Executable[Dict[str, Any], Dict[str, Any]], ABC, JSONParserMixin):
    """
    基础代理类，定义所有代理的通用接口。
    
    使用 Template Method 模式：
    - process() 定义标准流程：validate -> execute -> post_process
    - 子类实现 _execute() 方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.container = Container.instance()
        self.llm_client = None
        
        # Initialize Anthropic client
        api_key = self.config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None

        # 可选注入的统一 LLMClient（DI）
        try:
            from paperbot.infrastructure.llm.base import LLMClient

            try:
                self.llm_client = self.container.resolve(LLMClient)
            except Exception:
                self.llm_client = None
        except Exception:
            self.llm_client = None
        
        if not self.client:
            self.logger.warning("Anthropic API key not found. AI features will be disabled.")

    # ==================== Template Method Pattern ====================
    
    async def execute(self, *args, **kwargs) -> ExecutionResult[Dict[str, Any]]:
        """
        新的统一执行接口，返回 ExecutionResult。
        兼容旧版：内部仍调用 process()。
        """
        raw = await self.process(*args, **kwargs)
        return ensure_execution_result(raw)

    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Template Method: 定义标准处理流程。
        子类可以覆盖 _validate_input, _execute, _post_process 钩子。
        """
        # Step 1: Validate input
        validation_result = self._validate_input(*args, **kwargs)
        if validation_result is not None:
            return validation_result  # Early return on validation failure
        
        # Step 2: Execute core logic (abstract, must be implemented)
        try:
            result = await self._execute(*args, **kwargs)
        except Exception as e:
            self.log_error(e, {"context": "execute", "args": str(args)[:100]})
            return {"status": "error", "error": str(e)}
        
        # Step 3: Post-process result
        return self._post_process(result)
    
    def _validate_input(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        钩子方法：验证输入参数。
        返回 None 表示验证通过，返回 Dict 表示错误响应。
        子类可覆盖以实现自定义验证。
        """
        return None  # Default: no validation
    
    @abstractmethod
    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        抽象方法：执行核心业务逻辑。
        子类必须实现此方法。
        """
        pass
    
    def _post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        钩子方法：后处理结果。
        子类可覆盖以添加额外处理（如日志、缓存）。
        """
        if "status" not in result:
            result["status"] = "success"
        return result

    # ==================== Hook for structured parsing ====================
    def _parse_structured(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        钩子：子类可覆盖以做结构化解析/清洗。
        默认直接返回。
        """
        return raw

    def _on_failure(self, error: Exception) -> Dict[str, Any]:
        """
        钩子：统一处理失败，便于子类降级。
        """
        self.log_error(error, {"context": "on_failure"})
        return {"status": "error", "error": str(error)}
    
    # ==================== LLM Utilities ====================

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

    # ==================== Logging Utilities ====================

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
