"""
展示层 - CLI、报告生成、API。
"""

# CLI
try:
    from .cli import run_cli
except ImportError:
    run_cli = None

# Reports
from .reports import ReportWriter

__all__ = ["run_cli", "ReportWriter"]

