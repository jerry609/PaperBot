"""
节点基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from loguru import logger

from ..utils.json_parser import RobustJSONParser

if TYPE_CHECKING:
    from paperbot.infrastructure.llm.base import LLMClient


class BaseNode(ABC):
    """报告引擎节点基类。"""
    
    def __init__(self, llm_client: Optional["LLMClient"], name: str):
        """
        初始化节点。
        
        Args:
            llm_client: LLM 客户端（可选）
            name: 节点名称
        """
        self.llm = llm_client
        self.name = name
        self.parser = RobustJSONParser()

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:  # pragma: no cover - interface
        """执行节点逻辑。"""
        ...

    def info(self, msg: str):
        """记录信息日志。"""
        logger.info(f"[{self.name}] {msg}")

    def warn(self, msg: str):
        """记录警告日志。"""
        logger.warning(f"[{self.name}] {msg}")

