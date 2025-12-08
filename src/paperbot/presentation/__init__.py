"""
展示层 - CLI、报告生成、API。
"""

from .cli import run_cli
from .reports import ReportGenerator

__all__ = ["run_cli", "ReportGenerator"]

