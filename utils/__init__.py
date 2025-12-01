# securipaperbot/utils/__init__.py

from .logger import setup_logger, LogContext, log_with_context
from .downloader import PaperDownloader
from .retry_helper import (
    RetryConfig,
    with_retry,
    retry_on_network_error,
    with_graceful_retry,
    make_retryable_request,
    RetryableError,
    # 预定义配置
    LLM_RETRY_CONFIG,
    SEMANTIC_SCHOLAR_RETRY_CONFIG,
    GITHUB_RETRY_CONFIG,
    SEARCH_API_RETRY_CONFIG,
    DB_RETRY_CONFIG,
)
from .json_parser import (
    RobustJSONParser,
    JSONParseError,
    parse_json,
    safe_parse_json,
)

__all__ = [
    # 日志
    'setup_logger',
    'LogContext', 
    'log_with_context',
    # 下载
    'PaperDownloader',
    # 重试机制
    'RetryConfig',
    'with_retry',
    'retry_on_network_error',
    'with_graceful_retry',
    'make_retryable_request',
    'RetryableError',
    'LLM_RETRY_CONFIG',
    'SEMANTIC_SCHOLAR_RETRY_CONFIG',
    'GITHUB_RETRY_CONFIG',
    'SEARCH_API_RETRY_CONFIG',
    'DB_RETRY_CONFIG',
    # JSON 解析
    'RobustJSONParser',
    'JSONParseError',
    'parse_json',
    'safe_parse_json',
]