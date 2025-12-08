"""
渲染器基础定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Protocol


@dataclass
class RenderContext:
    """渲染上下文，包含渲染所需的所有数据。"""
    title: str
    subtitle: str
    toc: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    model_info: str = ""
    data_time: str = ""
    env_info: str = ""


class Renderer(Protocol):
    """渲染器协议。"""
    
    def render(self, ctx: RenderContext, **kwargs) -> Any:  # pragma: no cover - interface
        """
        渲染报告。
        
        Args:
            ctx: 渲染上下文
            **kwargs: 额外参数
            
        Returns:
            渲染结果
        """
        ...

