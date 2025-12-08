"""
依赖注入容器的注册入口，避免在各处手动 new 依赖。
"""

from __future__ import annotations

import os
from typing import Optional

from core.di.container import Container

try:
    from core.llm_client import LLMClient  # type: ignore
except Exception:  # pragma: no cover - 环境可能缺少 openai
    LLMClient = None


def bootstrap_dependencies(model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> Container:
    """
    注册全局依赖（目前主要是 LLMClient），如果缺少依赖则安全跳过。
    优先使用入参，其次读取环境变量。
    """
    container = Container.instance()

    llm_key = api_key or os.getenv("OPENAI_API_KEY")
    llm_model = model or os.getenv("OPENAI_MODEL") or os.getenv("PAPERBOT_HOST_MODEL")
    llm_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("PAPERBOT_HOST_BASE_URL")

    if LLMClient and llm_key and llm_model:
        def _factory():
            return LLMClient(api_key=llm_key, model_name=llm_model, base_url=llm_base_url)

        try:
            container.register(LLMClient, _factory, singleton=True)
        except Exception:
            # 如果注册失败，不影响主流程
            pass

    return container

