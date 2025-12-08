"""
通用执行抽象：用于统一 Agent/Node 的执行契约与结果结构。

- ExecutionResult: 标准化的 success/data/error/metadata 结果对象
- Executable: 具备 validate + execute 的基类接口
- ensure_execution_result: 将 dict/ExecutionResult 规范化，便于旧逻辑过渡
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar, Union, Callable

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass
class ExecutionResult(Generic[TOutput]):
    success: bool
    data: Optional[TOutput] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: TOutput, **metadata: Any) -> "ExecutionResult[TOutput]":
        """创建成功结果。"""
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "ExecutionResult[TOutput]":
        """创建失败结果。"""
        return cls(success=False, error=error, metadata=metadata or {})

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.duration_ms is None:
            return None
        return self.duration_ms / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，兼容旧的 status 风格。"""
        payload: Dict[str, Any] = {
            "success": self.success,
            "status": "success" if self.success else "error",
            "error": self.error,
            "data": self.data,
        }
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
            payload["duration_seconds"] = self.duration_seconds
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def map(self, fn: Callable[[TOutput], Any]) -> "ExecutionResult[Any]":
        """对数据进行映射，错误结果直接透传。"""
        if not self.success:
            return ExecutionResult(success=False, error=self.error, metadata=self.metadata)
        return ExecutionResult(success=True, data=fn(self.data), metadata=self.metadata)


class Executable(ABC, Generic[TInput, TOutput]):
    """
    可执行单元抽象。
    子类需实现 _execute，并可按需覆盖 validate/post_process。
    """

    def validate(self, input_data: TInput, **kwargs: Any) -> Optional[str]:
        """输入校验，不通过时返回错误消息。"""
        return None

    def post_process(self, result: ExecutionResult[TOutput]) -> ExecutionResult[TOutput]:
        """结果后处理，默认直接返回。"""
        return result

    async def __call__(self, input_data: TInput, **kwargs: Any) -> ExecutionResult[TOutput]:
        return await self.execute(input_data, **kwargs)

    async def execute(self, input_data: TInput, **kwargs: Any) -> ExecutionResult[TOutput]:
        """模板方法：validate -> _execute -> post_process。"""
        validation_error = self.validate(input_data, **kwargs)
        if validation_error:
            return ExecutionResult.fail(validation_error)

        try:
            raw = await self._execute(input_data, **kwargs)
            result = ensure_execution_result(raw)
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult.fail(str(exc))

        return self.post_process(result)

    @abstractmethod
    async def _execute(self, input_data: TInput, **kwargs: Any) -> Union[ExecutionResult[TOutput], TOutput, Dict[str, Any]]:
        """子类需实现核心执行逻辑。"""
        raise NotImplementedError


def ensure_execution_result(raw: Union[ExecutionResult[TOutput], Dict[str, Any], TOutput]) -> ExecutionResult[TOutput]:
    """
    将任意结果规范化为 ExecutionResult。
    - 如果是 ExecutionResult: 原样返回
    - 如果是 dict: 尝试解析 success/status/data/error
    - 其他: 视为成功数据
    """
    if isinstance(raw, ExecutionResult):
        return raw

    if isinstance(raw, dict):
        # 兼容旧的 status 字段
        success = raw.get("success")
        if success is None:
            status = raw.get("status")
            success = status == "success" if status is not None else True
        error = raw.get("error")
        data = raw.get("data")
        metadata = raw.get("metadata", {})
        duration_ms = raw.get("duration_ms")
        return ExecutionResult(
            success=bool(success),
            error=error,
            data=data,
            metadata=metadata if isinstance(metadata, dict) else {},
            duration_ms=duration_ms,
        )

    return ExecutionResult.ok(raw)  # type: ignore[arg-type]

