"""
外部 API 客户端。
"""

from .base import APIClient
from .semantic_scholar import SemanticScholarClient

__all__ = ["APIClient", "SemanticScholarClient"]

