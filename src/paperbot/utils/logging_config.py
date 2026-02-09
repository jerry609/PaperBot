# src/paperbot/utils/logging_config.py
"""
Centralized logging configuration for PaperBot.

Usage:
    from paperbot.utils.logging_config import Logger

    # Log to specific file with automatic line number
    Logger.info("Processing started", file="harvest/harvest.log")
    Logger.error("Failed to connect", file="errors/error.log")

    # Log to default file (logs/paperbot.log)
    Logger.info("General message")

Configuration via environment variables:
    PAPERBOT_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
    PAPERBOT_LOG_DIR: Base directory for log files (default: logs/)
    PAPERBOT_LOG_MAX_BYTES: Max size per log file in bytes (default: 10MB)
    PAPERBOT_LOG_BACKUP_COUNT: Number of backup files to keep (default: 5)
"""

from __future__ import annotations

import inspect
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Context variable for trace_id (thread-safe, async-safe)
_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

# Default configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "paperbot.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
DEFAULT_FORMAT = "{timestamp} [{level}] [{trace_id}] {filename}:{lineno} - {message}"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# Config file in the same directory as this module
LOG_CONFIG_FILE = Path(__file__).parent / "log_config.yaml"


class _LogFilesMeta(type):
    """Metaclass to allow attribute access like LogFiles.HARVEST."""

    def __getattr__(cls, name: str) -> str:
        cls._load()
        # Try exact match first, then case-insensitive
        if name in cls._files:
            return cls._files[name]
        key = name.lower()
        if key in cls._files:
            return cls._files[key]
        raise AttributeError(f"Log file '{name}' not found in config")


class LogFiles(metaclass=_LogFilesMeta):
    """
    Log file paths loaded from src/paperbot/utils/log_config.yaml.

    Usage:
        from paperbot.utils.logging_config import Logger, LogFiles

        Logger.info("Message", file=LogFiles.HARVEST)
        Logger.error("Error", file=LogFiles.ERROR)

    To add a new log file:
        1. Edit src/paperbot/utils/log_config.yaml
        2. Add entry under 'files' section
        3. Access via LogFiles.YOUR_NAME (uppercase)
    """

    _loaded = False
    _files: dict = {}

    @classmethod
    def _load(cls) -> None:
        """Load log file paths from config file."""
        if cls._loaded:
            return

        # Default values
        cls._files = {
            "harvest": "harvest/harvest.log",
            "api": "api/api.log",
            "error": "errors/error.log",
        }

        # Try to load from config file
        try:
            import yaml
            config_path = LOG_CONFIG_FILE  # Already a Path object
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config and "files" in config:
                        cls._files.update(config["files"])
        except Exception:
            pass

        cls._loaded = True

    @classmethod
    def get(cls, name: str) -> str:
        """Get log file path by name."""
        cls._load()
        # Try exact match first, then case-insensitive
        if name in cls._files:
            return cls._files[name]
        key = name.lower()
        if key in cls._files:
            return cls._files[key]
        return f"{name}/{name}.log"

# Log levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# Module-level state
_initialized = False
_config: dict = {}
_file_handlers: dict[str, RotatingFileHandler] = {}


def _get_config() -> dict:
    """Get logging configuration from environment variables."""
    return {
        "level": os.environ.get("PAPERBOT_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        "base_dir": os.environ.get("PAPERBOT_LOG_DIR", DEFAULT_LOG_DIR),
        "max_bytes": int(os.environ.get("PAPERBOT_LOG_MAX_BYTES", DEFAULT_MAX_BYTES)),
        "backup_count": int(os.environ.get("PAPERBOT_LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT)),
    }


def _ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _get_file_handler(file_path: str) -> RotatingFileHandler:
    """Get or create a file handler for the given path."""
    if file_path not in _file_handlers:
        path = Path(file_path)
        _ensure_dir(path.parent)

        handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=_config.get("max_bytes", DEFAULT_MAX_BYTES),
            backupCount=_config.get("backup_count", DEFAULT_BACKUP_COUNT),
            encoding="utf-8",
        )
        _file_handlers[file_path] = handler

    return _file_handlers[file_path]


def _format_message(
    level: str,
    message: str,
    filename: str,
    lineno: int,
    trace_id: Optional[str] = None,
) -> str:
    """Format a log message."""
    timestamp = datetime.now().strftime(DEFAULT_DATE_FORMAT)
    tid = trace_id or _trace_id_var.get() or "-"
    return DEFAULT_FORMAT.format(
        timestamp=timestamp,
        level=level,
        trace_id=tid,
        filename=filename,
        lineno=lineno,
        message=message,
    )


def _resolve_file_path(file: Optional[str]) -> str:
    """Resolve the full file path for logging."""
    base_dir = _config.get("base_dir", DEFAULT_LOG_DIR)

    if file is None:
        return str(Path(base_dir) / DEFAULT_LOG_FILE)

    # If file contains directory separator, use as relative path under base_dir
    if "/" in file or "\\" in file:
        return str(Path(base_dir) / file)

    # Otherwise, just a filename in base_dir
    return str(Path(base_dir) / file)


def _should_log(level: str) -> bool:
    """Check if message should be logged based on current level."""
    current_level = _config.get("level", DEFAULT_LOG_LEVEL)
    return LOG_LEVELS.get(level, 0) >= LOG_LEVELS.get(current_level, 0)


def _write_log(level: str, message: str, file: Optional[str] = None) -> None:
    """Write a log message to the specified file."""
    if not _should_log(level):
        return

    # Get caller info (skip _write_log and the public method)
    frame = inspect.currentframe()
    caller_frame = frame.f_back.f_back if frame and frame.f_back else None

    if caller_frame:
        filename = os.path.basename(caller_frame.f_code.co_filename)
        lineno = caller_frame.f_lineno
    else:
        filename = "unknown"
        lineno = 0

    # Format message (trace_id is automatically obtained from context)
    formatted = _format_message(level, message, filename, lineno)

    # Get file path and handler
    file_path = _resolve_file_path(file)
    handler = _get_file_handler(file_path)

    # Write to file
    handler.stream.write(formatted + "\n")
    handler.stream.flush()


class Logger:
    """
    Static logger class for logging to specific files.

    Usage:
        from paperbot.utils.logging_config import Logger

        # Initialize once at application startup (optional, auto-initializes on first use)
        Logger.init()

        # Log to specific file
        Logger.info("Processing started", file="harvest/harvest.log")
        Logger.error("Failed", file="errors/error.log")

        # Log to default file (logs/paperbot.log)
        Logger.info("General message")
    """

    @staticmethod
    def init(
        level: Optional[str] = None,
        base_dir: Optional[str] = None,
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
    ) -> None:
        """
        Initialize the logging system.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            base_dir: Base directory for all log files
            max_bytes: Maximum size of each log file before rotation
            backup_count: Number of backup files to keep
        """
        global _initialized, _config

        if _initialized:
            return

        _config = _get_config()

        if level:
            _config["level"] = level.upper()
        if base_dir:
            _config["base_dir"] = base_dir
        if max_bytes:
            _config["max_bytes"] = max_bytes
        if backup_count:
            _config["backup_count"] = backup_count

        _initialized = True

    @staticmethod
    def _ensure_init() -> None:
        """Ensure logger is initialized."""
        if not _initialized:
            Logger.init()

    @staticmethod
    def debug(message: str, file: Optional[str] = None) -> None:
        """Log a debug message."""
        Logger._ensure_init()
        _write_log("DEBUG", message, file)

    @staticmethod
    def info(message: str, file: Optional[str] = None) -> None:
        """Log an info message."""
        Logger._ensure_init()
        _write_log("INFO", message, file)

    @staticmethod
    def warning(message: str, file: Optional[str] = None) -> None:
        """Log a warning message."""
        Logger._ensure_init()
        _write_log("WARNING", message, file)

    @staticmethod
    def error(message: str, file: Optional[str] = None) -> None:
        """Log an error message."""
        Logger._ensure_init()
        _write_log("ERROR", message, file)

    @staticmethod
    def critical(message: str, file: Optional[str] = None) -> None:
        """Log a critical message."""
        Logger._ensure_init()
        _write_log("CRITICAL", message, file)

    @staticmethod
    def set_level(level: str) -> None:
        """Change the log level at runtime."""
        Logger._ensure_init()
        _config["level"] = level.upper()

    @staticmethod
    def close() -> None:
        """Close all file handlers."""
        for handler in _file_handlers.values():
            handler.close()
        _file_handlers.clear()


# ============================================================================
# Trace ID Management
# ============================================================================

def generate_trace_id() -> str:
    """Generate a new trace ID."""
    return f"req-{uuid.uuid4().hex[:12]}"


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """
    Set the trace ID for the current context.

    If no trace_id is provided, generates a new one.
    Returns the trace_id that was set.

    Usage:
        # At the start of a request handler
        trace_id = set_trace_id()
        Logger.info("Request started", file=LogFiles.HARVEST)  # auto-includes trace_id
    """
    tid = trace_id or generate_trace_id()
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    return _trace_id_var.get()


def clear_trace_id() -> None:
    """Clear the current trace ID."""
    _trace_id_var.set(None)
