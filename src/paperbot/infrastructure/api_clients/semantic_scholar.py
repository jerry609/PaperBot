"""
Semantic Scholar API 客户端
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

from .base import APIClient

logger = logging.getLogger(__name__)


class SemanticScholarClient(APIClient):
    """Semantic Scholar API 专用客户端"""

    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        request_interval: float = 1.0,
    ):
        super().__init__(
            base_url=self.S2_API_BASE,
            api_key=api_key,
            timeout=timeout,
            request_interval=request_interval,
        )

    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索论文

        Args:
            query: 搜索关键词
            limit: 最大结果数
            fields: 要返回的字段列表

        Returns:
            论文列表
        """
        if fields is None:
            fields = ["title", "abstract", "year", "citationCount", "authors"]

        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields),
        }

        try:
            data = await self.get("paper/search", params=params)
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"S2 search error: {e}")
            return []

    async def search_authors(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索作者

        Args:
            query: 作者关键词
            limit: 最大结果数
            fields: 要返回的字段列表

        Returns:
            作者列表
        """
        if fields is None:
            fields = ["name", "affiliations", "paperCount", "citationCount", "hIndex"]

        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields),
        }

        try:
            data = await self.get("author/search", params=params)
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"S2 author search error: {e}")
            return []

    async def get_paper(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        获取论文详情

        Args:
            paper_id: 论文 ID 或 DOI
            fields: 要返回的字段列表

        Returns:
            论文详情或 None
        """
        if fields is None:
            fields = [
                "title",
                "abstract",
                "year",
                "citationCount",
                "authors",
                "references",
                "citations",
            ]

        params = {"fields": ",".join(fields)}

        try:
            return await self.get(f"paper/{paper_id}", params=params)
        except Exception as e:
            logger.warning(f"S2 paper lookup error: {e}")
            return None

    async def get_author(
        self,
        author_id: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        获取作者信息

        Args:
            author_id: 作者 ID
            fields: 要返回的字段列表

        Returns:
            作者信息或 None
        """
        if fields is None:
            fields = ["name", "affiliations", "paperCount", "citationCount", "hIndex"]

        params = {"fields": ",".join(fields)}

        try:
            return await self.get(f"author/{author_id}", params=params)
        except Exception as e:
            logger.warning(f"S2 author lookup error: {e}")
            return None

    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取作者的论文列表

        Args:
            author_id: 作者 ID
            limit: 最大结果数
            fields: 要返回的字段列表

        Returns:
            论文列表
        """
        if fields is None:
            fields = ["title", "year", "citationCount", "venue"]

        params = {
            "fields": ",".join(fields),
            "limit": limit,
        }

        try:
            data = await self.get(f"author/{author_id}/papers", params=params)
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"S2 author papers error: {e}")
            return []
