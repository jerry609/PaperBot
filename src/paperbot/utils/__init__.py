# paperbot/utils/__init__.py
"""
PaperBot 工具函数模块

包含:
- logger: 日志配置
- downloader: 论文下载器
- retry_helper: 重试机制
- json_parser: JSON 解析
- text_processing: 文本处理
- search: 搜索工具
- analyzer: 分析工具
- conference_*: 会议相关工具
"""

from paperbot.utils.logger import setup_logger, LogContext, log_with_context
from paperbot.utils.downloader import PaperDownloader
from paperbot.utils.retry_helper import (
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
from paperbot.utils.json_parser import (
    RobustJSONParser,
    JSONParseError,
    parse_json,
    safe_parse_json,
)
from paperbot.utils.text_processing import (
    clean_json_tags,
    remove_reasoning_from_output,
    extract_clean_response,
    fix_incomplete_json,
    truncate_content,
    format_search_results_for_prompt,
    # 学者追踪专用
    format_paper_for_prompt,
    format_scholar_for_prompt,
    deduplicate_papers,
    extract_github_url,
    extract_arxiv_id,
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
    # 文本处理
    'clean_json_tags',
    'remove_reasoning_from_output',
    'extract_clean_response',
    'fix_incomplete_json',
    'truncate_content',
    'format_search_results_for_prompt',
    'format_paper_for_prompt',
    'format_scholar_for_prompt',
    'deduplicate_papers',
    'extract_github_url',
    'extract_arxiv_id',
]
