"""
统一错误与 Result 封装，便于流水线按严重级别降级或中断。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, TypeVar, Union, cast


class ErrorSeverity(Enum):
    WARNING = "warning"      # 可继续
    ERROR = "error"          # 阶段失败
    CRITICAL = "critical"    # 流水线终止


@dataclass
class PaperBotError(Exception):
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    code: str = "UNKNOWN"
    context: Dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class LLMError(PaperBotError):
    code = "LLM_ERROR"


class APIError(PaperBotError):
    code = "API_ERROR"


class ValidationError(PaperBotError):
    code = "VALIDATION_ERROR"
    severity = ErrorSeverity.WARNING


T = TypeVar("T")
E = TypeVar("E", bound=PaperBotError)


@dataclass
class Result(Generic[T, E]):
    """函数式结果封装，避免散落的 status 字典。"""

    _value: Union[T, E]
    _is_ok: bool

    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        return cls(_value=value, _is_ok=True)

    @classmethod
    def err(cls, error: E) -> "Result[T, E]":
        return cls(_value=error, _is_ok=False)

    def is_ok(self) -> bool:
        return self._is_ok

    def unwrap(self) -> T:
        if not self._is_ok:
            raise cast(E, self._value)
        return cast(T, self._value)

    def unwrap_or(self, default: T) -> T:
        return cast(T, self._value) if self._is_ok else default

    def map(self, fn) -> "Result[T, E]":
        if self._is_ok:
            return Result.ok(fn(self._value))  # type: ignore[arg-type]
        return self

