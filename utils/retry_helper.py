"""
重试机制工具模块
提供通用的网络请求重试功能，增强系统健壮性

来源: BettaFish/utils/retry_helper.py
适配: PaperBot 学者追踪系统
"""

import time
from functools import wraps
from typing import Callable, Any, Optional, Tuple
import requests
from loguru import logger


class RetryConfig:
    """重试配置类"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        retry_on_exceptions: Optional[Tuple] = None
    ):
        """
        初始化重试配置
        
        Args:
            max_retries: 最大重试次数
            initial_delay: 初始延迟秒数
            backoff_factor: 退避因子（每次重试延迟翻倍）
            max_delay: 最大延迟秒数
            retry_on_exceptions: 需要重试的异常类型元组
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        
        # 默认需要重试的异常类型
        if retry_on_exceptions is None:
            self.retry_on_exceptions = (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.TooManyRedirects,
                ConnectionError,
                TimeoutError,
            )
        else:
            self.retry_on_exceptions = retry_on_exceptions


# 默认配置
DEFAULT_RETRY_CONFIG = RetryConfig()


def with_retry(config: Optional[RetryConfig] = None):
    """
    重试装饰器
    
    Args:
        config: 重试配置，如果不提供则使用默认配置
    
    Returns:
        装饰器函数
    
    Example:
        @with_retry(LLM_RETRY_CONFIG)
        def call_api():
            return requests.get("https://api.example.com")
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"函数 {func.__name__} 在第 {attempt + 1} 次尝试后成功")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        logger.error(f"函数 {func.__name__} 在 {config.max_retries + 1} 次尝试后仍然失败")
                        logger.error(f"最终错误: {str(e)}")
                        raise e
                    
                    # 计算延迟时间（指数退避）
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}")
                    logger.info(f"将在 {delay:.1f} 秒后进行第 {attempt + 2} 次尝试...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    logger.error(f"函数 {func.__name__} 遇到不可重试的异常: {str(e)}")
                    raise e
            
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def retry_on_network_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    专门用于网络错误的重试装饰器（简化版）
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟秒数
        backoff_factor: 退避因子
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    return with_retry(config)


class RetryableError(Exception):
    """自定义的可重试异常"""
    pass


def with_graceful_retry(config: Optional[RetryConfig] = None, default_return=None):
    """
    优雅重试装饰器 - 用于非关键API调用
    失败后不会抛出异常，而是返回默认值，保证系统继续运行
    
    Args:
        config: 重试配置，如果不提供则使用默认配置
        default_return: 所有重试失败后返回的默认值
    
    Example:
        @with_graceful_retry(default_return=[])
        def fetch_optional_data():
            return requests.get("https://api.example.com").json()
    """
    if config is None:
        config = SEMANTIC_SCHOLAR_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"非关键API {func.__name__} 在第 {attempt + 1} 次尝试后成功")
                    return result
                    
                except config.retry_on_exceptions as e:
                    if attempt == config.max_retries:
                        logger.warning(f"非关键API {func.__name__} 在 {config.max_retries + 1} 次尝试后仍然失败")
                        logger.warning(f"最终错误: {str(e)}")
                        logger.info(f"返回默认值以保证系统继续运行: {default_return}")
                        return default_return
                    
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"非关键API {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}")
                    logger.info(f"将在 {delay:.1f} 秒后进行第 {attempt + 2} 次尝试...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    logger.warning(f"非关键API {func.__name__} 遇到不可重试的异常: {str(e)}")
                    logger.info(f"返回默认值以保证系统继续运行: {default_return}")
                    return default_return
            
            return default_return
            
        return wrapper
    return decorator


def make_retryable_request(
    request_func: Callable,
    *args,
    max_retries: int = 5,
    **kwargs
) -> Any:
    """
    直接执行可重试的请求（不使用装饰器）
    
    Args:
        request_func: 要执行的请求函数
        *args: 传递给请求函数的位置参数
        max_retries: 最大重试次数
        **kwargs: 传递给请求函数的关键字参数
    """
    config = RetryConfig(max_retries=max_retries)
    
    @with_retry(config)
    def _execute():
        return request_func(*args, **kwargs)
    
    return _execute()


# ============ 预定义的重试配置 ============

# LLM API 重试配置（长等待）
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=6,
    initial_delay=60.0,
    backoff_factor=2.0,
    max_delay=600.0
)

# Semantic Scholar API 重试配置
SEMANTIC_SCHOLAR_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    backoff_factor=1.6,
    max_delay=30.0
)

# GitHub API 重试配置
GITHUB_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=15.0
)

# 通用搜索 API 重试配置
SEARCH_API_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    backoff_factor=1.6,
    max_delay=25.0
)

# 数据库操作重试配置
DB_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=1.0,
    backoff_factor=1.5,
    max_delay=10.0
)
