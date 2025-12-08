"""
LLM 模型选择策略。

提供分层模型选择能力，不同任务可使用不同模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LLMStrategy:
    """
    简单的分层模型选择策略。
    task_type -> model_name
    """
    default_model: str
    task_models: Dict[str, str] = field(default_factory=dict)

    def pick(self, task_type: str) -> str:
        """根据任务类型选择模型。"""
        return self.task_models.get(task_type, self.default_model)

