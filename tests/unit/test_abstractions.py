"""
核心抽象层单元测试
"""

import pytest
import asyncio
from typing import Dict, Any

# 尝试从新路径导入，降级到旧路径
try:
    from src.paperbot.core.abstractions import (
        Executable,
        ExecutionResult,
        ensure_execution_result,
    )
except ImportError:
    from core.abstractions import (
        Executable,
        ExecutionResult,
        ensure_execution_result,
    )


class TestExecutionResult:
    """ExecutionResult 测试"""
    
    def test_ok_creates_success_result(self):
        result = ExecutionResult.ok({"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
    
    def test_fail_creates_error_result(self):
        result = ExecutionResult.fail("Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None
    
    def test_to_dict_includes_status(self):
        result = ExecutionResult.ok({"test": 1})
        d = result.to_dict()
        assert d["success"] is True
        assert d["status"] == "success"
        assert d["data"] == {"test": 1}
    
    def test_map_transforms_data(self):
        result = ExecutionResult.ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.data == 10
    
    def test_map_preserves_error(self):
        result = ExecutionResult.fail("error")
        mapped = result.map(lambda x: x * 2)
        assert mapped.success is False
        assert mapped.error == "error"


class TestEnsureExecutionResult:
    """ensure_execution_result 测试"""
    
    def test_passthrough_execution_result(self):
        original = ExecutionResult.ok("data")
        result = ensure_execution_result(original)
        assert result is original
    
    def test_convert_dict_with_success(self):
        d = {"success": True, "data": "value"}
        result = ensure_execution_result(d)
        assert result.success is True
        assert result.data == "value"
    
    def test_convert_dict_with_status(self):
        d = {"status": "success", "data": "value"}
        result = ensure_execution_result(d)
        assert result.success is True
    
    def test_convert_dict_with_error_status(self):
        d = {"status": "error", "error": "failed"}
        result = ensure_execution_result(d)
        assert result.success is False
        assert result.error == "failed"
    
    def test_convert_raw_value(self):
        result = ensure_execution_result("raw string")
        assert result.success is True
        assert result.data == "raw string"


class DummyExecutable(Executable[Dict[str, Any], Dict[str, Any]]):
    """测试用的 Executable 实现"""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
    
    async def _execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self.should_fail:
            raise ValueError("Intentional failure")
        return {"processed": input_data.get("value", 0) * 2}


class TestExecutable:
    """Executable 基类测试"""
    
    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        exe = DummyExecutable()
        result = await exe.execute({"value": 5})
        assert result.success is True
        assert result.data["processed"] == 10
    
    @pytest.mark.asyncio
    async def test_execute_catches_exception(self):
        exe = DummyExecutable(should_fail=True)
        result = await exe.execute({"value": 5})
        assert result.success is False
        assert "Intentional failure" in result.error
    
    @pytest.mark.asyncio
    async def test_callable_interface(self):
        exe = DummyExecutable()
        result = await exe({"value": 3})
        assert result.success is True
        assert result.data["processed"] == 6

