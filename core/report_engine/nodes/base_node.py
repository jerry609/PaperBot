"""
节点基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from ..utils.json_parser import RobustJSONParser
from ...llm_client import LLMClient


class BaseNode(ABC):
    def __init__(self, llm_client: LLMClient, name: str):
        self.llm = llm_client
        self.name = name
        self.parser = RobustJSONParser()

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:  # pragma: no cover - interface
        ...

    def info(self, msg: str):
        logger.info(f"[{self.name}] {msg}")

    def warn(self, msg: str):
        logger.warning(f"[{self.name}] {msg}")

