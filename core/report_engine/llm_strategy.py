from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LLMStrategy:
    """
    简单的分层模型选择策略。
    task_type -> model_name
    """
    default_model: str
    task_models: Dict[str, str]

    def pick(self, task_type: str) -> str:
        return self.task_models.get(task_type, self.default_model)

