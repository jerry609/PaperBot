# securipaperbot/utils/logger.py

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
from logging.handlers import RotatingFileHandler


class AnalysisFormatter(logging.Formatter):
    """自定义日志格式化器"""

    def format(self, record):
        """格式化日志记录"""
        # 添加时间戳
        record.timestamp = datetime.now().isoformat()

        # 添加上下文信息
        if hasattr(record, 'context') and record.context:
            record.context_str = json.dumps(record.context)
        else:
            record.context_str = ''

        return super().format(record)


def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
) -> logging.Logger:
    """设置日志记录器"""

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建格式化器
    formatter = AnalysisFormatter(
        '%(timestamp)s [%(name)s] %(levelname)s: %(message)s%(context_str)s'
    )

    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建循环文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class LogContext:
    """日志上下文管理器"""

    def __init__(self, logger: logging.Logger, context: dict):
        self.logger = logger
        self.context = context
        self.original_context = None

    def __enter__(self):
        """进入上下文"""
        # 保存原始上下文
        self.original_context = getattr(self.logger, 'context', {})
        # 设置新上下文
        self.logger.context = self.context
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        # 恢复原始上下文
        self.logger.context = self.original_context


def log_with_context(logger: logging.Logger, level: int, message: str, context: dict = None):
    """带上下文的日志记录"""
    if context:
        with LogContext(logger, context):
            logger.log(level, message)
    else:
        logger.log(level, message)


def get_error_context(error: Exception) -> dict:
    """获取异常的上下文信息"""
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': getattr(error, '__traceback__', None)
    }


# 示例使用
if __name__ == '__main__':
    # 设置日志
    logger = setup_logger(
        'securipaperbot',
        log_file='logs/analysis.log'
    )

    # 基本日志
    logger.info("Starting analysis")

    # 带上下文的日志
    context = {
        'conference': 'CCS',
        'year': '2023',
        'paper_count': 50
    }
    log_with_context(logger, logging.INFO, "Processing papers", context)

    # 错误日志
    try:
        raise ValueError("Invalid paper format")
    except Exception as e:
        error_context = get_error_context(e)
        log_with_context(logger, logging.ERROR, "Analysis failed", error_context)