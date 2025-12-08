"""
依赖注入模块。
"""

from .container import Container, inject
from .bootstrap import bootstrap_dependencies

__all__ = ["Container", "inject", "bootstrap_dependencies"]

