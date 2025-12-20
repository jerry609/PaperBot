# src/paperbot/agents/huggingface/__init__.py
"""
HuggingFace Agent 模块

从 HuggingFace Hub 获取论文关联模型的元数据。
"""

from .agent import HuggingFaceAgent

__all__ = ["HuggingFaceAgent"]
