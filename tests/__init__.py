import sys
from unittest.mock import MagicMock

# Mock dependencies globally for tests
# Core AI/ML
sys.modules["pytest"] = MagicMock()
sys.modules["docker"] = MagicMock()
sys.modules["docker.errors"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["claude_agent_sdk"] = MagicMock()

# Git/GitHub
sys.modules["git"] = MagicMock()
sys.modules["github"] = MagicMock()

# Web scraping
sys.modules["bs4"] = MagicMock()
sys.modules["lxml"] = MagicMock()

# Logging
sys.modules["loguru"] = MagicMock()
sys.modules["colorlog"] = MagicMock()

# Code analysis
sys.modules["radon"] = MagicMock()
sys.modules["radon.complexity"] = MagicMock()
sys.modules["radon.visitors"] = MagicMock()
sys.modules["safety"] = MagicMock()

# PDF processing
sys.modules["pdfplumber"] = MagicMock()
sys.modules["PyPDF2"] = MagicMock()

# Async/HTTP
sys.modules["aiohttp"] = MagicMock()
sys.modules["aiofiles"] = MagicMock()
sys.modules["httpx"] = MagicMock()

# Templating
sys.modules["jinja2"] = MagicMock()
sys.modules["markdown"] = MagicMock()

# Data processing
sys.modules["pandas"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["yaml"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["python_dotenv"] = MagicMock()

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
