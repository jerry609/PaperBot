"""
错误处理单元测试
"""

import pytest

try:
    from src.paperbot.core.errors import (
        ErrorSeverity,
        PaperBotError,
        LLMError,
        APIError,
        ValidationError,
        Result,
    )
except ImportError:
    from core.errors import (
        ErrorSeverity,
        PaperBotError,
        LLMError,
        APIError,
        ValidationError,
        Result,
    )


class TestErrorSeverity:
    """ErrorSeverity 测试"""
    
    def test_severity_values(self):
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestPaperBotError:
    """PaperBotError 测试"""
    
    def test_error_str(self):
        err = PaperBotError(message="Test error", code="TEST")
        assert "[TEST] Test error" in str(err)
    
    def test_error_with_context(self):
        err = PaperBotError(
            message="Failed",
            context={"key": "value"},
        )
        assert err.context == {"key": "value"}


class TestSpecificErrors:
    """特定错误类型测试"""
    
    def test_llm_error(self):
        err = LLMError(message="API timeout")
        assert err.code == "LLM_ERROR"
    
    def test_api_error(self):
        err = APIError(message="Rate limited")
        assert err.code == "API_ERROR"
    
    def test_validation_error_is_warning(self):
        err = ValidationError(message="Invalid input")
        assert err.severity == ErrorSeverity.WARNING


class TestResult:
    """Result 类型测试"""
    
    def test_ok_result(self):
        result = Result.ok(42)
        assert result.is_ok() is True
        assert result.unwrap() == 42
    
    def test_err_result(self):
        err = PaperBotError(message="Failed")
        result = Result.err(err)
        assert result.is_ok() is False
        
        with pytest.raises(PaperBotError):
            result.unwrap()
    
    def test_unwrap_or_returns_default(self):
        err = PaperBotError(message="Failed")
        result = Result.err(err)
        assert result.unwrap_or("default") == "default"
    
    def test_unwrap_or_returns_value(self):
        result = Result.ok("actual")
        assert result.unwrap_or("default") == "actual"
    
    def test_map_transforms_ok(self):
        result = Result.ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.unwrap() == 10
    
    def test_map_preserves_err(self):
        err = PaperBotError(message="Failed")
        result = Result.err(err)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok() is False

