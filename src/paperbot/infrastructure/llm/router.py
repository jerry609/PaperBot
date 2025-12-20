# src/paperbot/infrastructure/llm/router.py
"""
Model Router - 模型路由器

根据任务类型智能选择最优的 LLM Provider。

设计原则:
- 简单任务 (实体提取) -> 便宜模型 (4o-mini)
- 复杂任务 (推理分析) -> 强模型 (Claude 3.5)
- 开发/离线 -> 本地模型 (Ollama)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Callable
from enum import Enum

from .providers.base import LLMProvider

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """任务类型枚举"""
    DEFAULT = "default"
    EXTRACTION = "extraction"      # 实体/关键词提取
    SUMMARY = "summary"            # 摘要生成
    ANALYSIS = "analysis"          # 深度分析
    REASONING = "reasoning"        # 复杂推理
    CODE = "code"                  # 代码生成/分析
    REVIEW = "review"              # 论文评审


@dataclass
class ModelConfig:
    """
    模型配置
    
    Attributes:
        provider: 提供商类型 (openai/anthropic/ollama)
        model: 模型名称
        cost_tier: 成本层级 (0=free, 1=cheap, 2=medium, 3=expensive)
        api_key_env: API Key 环境变量名
        base_url: 自定义 API 地址 (可选)
        max_tokens: 最大输出 tokens
    """
    provider: str
    model: str
    cost_tier: int = 1
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    max_tokens: int = 4096


@dataclass
class RouterConfig:
    """
    路由器配置
    
    Attributes:
        models: 任务类型 -> ModelConfig 映射
        fallback_model: 默认回退模型
    """
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    fallback_model: str = "default"


class ModelRouter:
    """
    智能模型路由器
    
    功能:
    - 根据任务类型选择最优模型
    - 缓存已创建的 Provider 实例
    - 支持运行时切换模型
    
    使用示例:
    ```python
    router = ModelRouter.from_env()
    
    # 获取适合实体提取的 provider
    provider = router.get_provider("extraction")
    result = provider.invoke_simple("Extract...", "Text...")
    
    # 获取适合复杂推理的 provider
    reasoning_provider = router.get_provider("reasoning")
    ```
    """
    
    # 默认任务路由
    DEFAULT_ROUTING = {
        TaskType.DEFAULT: "default",
        TaskType.EXTRACTION: "default",
        TaskType.SUMMARY: "default",
        TaskType.ANALYSIS: "reasoning",
        TaskType.REASONING: "reasoning",
        TaskType.CODE: "code",
        TaskType.REVIEW: "reasoning",
    }
    
    def __init__(self, config: RouterConfig):
        """
        初始化路由器
        
        Args:
            config: 路由器配置
        """
        self.config = config
        self._providers: Dict[str, LLMProvider] = {}
        self._task_routing: Dict[str, str] = dict(self.DEFAULT_ROUTING)
        logger.info(f"ModelRouter 初始化: {len(config.models)} 个模型配置")
    
    def get_provider(self, task_type: str = "default") -> LLMProvider:
        """
        获取适合指定任务的 Provider
        
        Args:
            task_type: 任务类型
            
        Returns:
            LLMProvider 实例
        """
        # 查找任务对应的模型配置名
        config_name = self._task_routing.get(task_type, self.config.fallback_model)
        
        # 检查缓存
        if config_name in self._providers:
            return self._providers[config_name]
        
        # 创建新 Provider
        model_config = self.config.models.get(config_name)
        if not model_config:
            # 尝试回退
            model_config = self.config.models.get(self.config.fallback_model)
            if not model_config:
                raise ValueError(f"未找到模型配置: {config_name}")
        
        provider = self._create_provider(model_config)
        self._providers[config_name] = provider
        
        return provider
    
    def _create_provider(self, config: ModelConfig) -> LLMProvider:
        """根据配置创建 Provider"""
        api_key = os.getenv(config.api_key_env, "")
        
        if config.provider == "openai":
            from .providers.openai_provider import OpenAIProvider
            return OpenAIProvider(
                api_key=api_key,
                model_name=config.model,
                base_url=config.base_url,
                cost_tier=config.cost_tier,
            )
        
        elif config.provider == "anthropic":
            from .providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(
                api_key=api_key,
                model_name=config.model,
                max_tokens=config.max_tokens,
                cost_tier=config.cost_tier,
            )
        
        elif config.provider == "ollama":
            from .providers.ollama_provider import OllamaProvider
            return OllamaProvider(
                model_name=config.model,
                base_url=config.base_url or "http://localhost:11434",
            )
        
        else:
            raise ValueError(f"未知的 provider 类型: {config.provider}")
    
    def set_task_routing(self, task_type: str, config_name: str) -> None:
        """设置任务路由"""
        self._task_routing[task_type] = config_name
    
    def list_models(self) -> Dict[str, str]:
        """列出所有配置的模型"""
        return {
            name: f"{cfg.provider}:{cfg.model}"
            for name, cfg in self.config.models.items()
        }
    
    @classmethod
    def from_env(cls) -> "ModelRouter":
        """
        从环境变量创建默认路由器
        
        环境变量:
        - OPENAI_API_KEY: OpenAI/DeepSeek API Key
        - ANTHROPIC_API_KEY: Claude API Key
        - LLM_DEFAULT_MODEL: 默认模型 (default: gpt-4o-mini)
        - LLM_REASONING_MODEL: 推理模型 (default: claude-3-5-sonnet-20241022)
        """
        default_model = os.getenv("LLM_DEFAULT_MODEL", "gpt-4o-mini")
        reasoning_model = os.getenv("LLM_REASONING_MODEL", "claude-3-5-sonnet-20241022")
        
        # 检测推理模型的 provider
        reasoning_provider = "anthropic" if "claude" in reasoning_model.lower() else "openai"
        reasoning_key_env = "ANTHROPIC_API_KEY" if reasoning_provider == "anthropic" else "OPENAI_API_KEY"
        
        config = RouterConfig(
            models={
                "default": ModelConfig(
                    provider="openai",
                    model=default_model,
                    cost_tier=1,
                    api_key_env="OPENAI_API_KEY",
                ),
                "reasoning": ModelConfig(
                    provider=reasoning_provider,
                    model=reasoning_model,
                    cost_tier=3,
                    api_key_env=reasoning_key_env,
                ),
                "code": ModelConfig(
                    provider="openai",
                    model="gpt-4o",
                    cost_tier=2,
                    api_key_env="OPENAI_API_KEY",
                ),
            },
            fallback_model="default",
        )
        
        return cls(config)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelRouter":
        """从 YAML 配置文件创建路由器"""
        import yaml
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        models = {}
        for name, cfg in data.get("providers", {}).items():
            models[name] = ModelConfig(
                provider=cfg.get("provider", "openai"),
                model=cfg.get("model", "gpt-4o-mini"),
                cost_tier=cfg.get("cost_tier", 1),
                api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                base_url=cfg.get("base_url"),
                max_tokens=cfg.get("max_tokens", 4096),
            )
        
        config = RouterConfig(
            models=models,
            fallback_model=data.get("fallback", "default"),
        )
        
        router = cls(config)
        
        # 应用任务路由
        for task, model_name in data.get("task_routing", {}).items():
            router.set_task_routing(task, model_name)
        
        return router
