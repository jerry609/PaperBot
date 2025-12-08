import sys
from unittest.mock import MagicMock

# Mock dependencies globally for tests
sys.modules["pytest"] = MagicMock()
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["claude_agent_sdk"] = MagicMock()

from .test_framework import (
    # Mock 数据生成器
    MockDataGenerator,
    # 测试基类
    BaseTestCase,
    AsyncTestCase,
    # 辅助函数
    assert_valid_scholar,
    assert_valid_paper,
    assert_valid_influence,
    load_test_data,
    save_test_output,
    # 常量
    TEST_DATA_DIR,
    TEST_OUTPUT_DIR,
)

__all__ = [
    "MockDataGenerator",
    "BaseTestCase",
    "AsyncTestCase",
    "assert_valid_scholar",
    "assert_valid_paper",
    "assert_valid_influence",
    "load_test_data",
    "save_test_output",
    "TEST_DATA_DIR",
    "TEST_OUTPUT_DIR",
]
