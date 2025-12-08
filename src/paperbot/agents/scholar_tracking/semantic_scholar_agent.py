# scholar_tracking/agents/semantic_scholar_agent.py
"""
Semantic Scholar API Agent
负责与 Semantic Scholar API 交互，获取学者和论文信息
"""

import logging
from typing import List, Dict, Any, Optional

from paperbot.agents.base import BaseAgent
from paperbot.domain.paper import PaperMeta
from paperbot.domain.scholar import Scholar
from paperbot.infrastructure.services.api_client import APIClient

logger = logging.getLogger(__name__)


class SemanticScholarAgent(BaseAgent):
    """
    Semantic Scholar API 封装 Agent
    
    主要功能：
    - 获取学者信息
    - 获取学者的论文列表
    - 获取论文详情
    """
    
    # API 基础 URL
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # 论文字段列表
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "venue",
        "citationCount",
        "influentialCitationCount",
        "fieldsOfStudy",
        "authors",
        "externalIds",
        "url",
        "openAccessPdf",
        "publicationDate",
    ]
    
    # 作者字段列表
    AUTHOR_FIELDS = [
        "authorId",
        "name",
        "affiliations",
        "homepage",
        "paperCount",
        "citationCount",
        "hIndex",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 从配置中获取 API 设置
        api_config = {}
        if config:
            api_config = config.get("api", {}).get("semantic_scholar", {})
        
        self.api_key = api_config.get("api_key")
        self.timeout = api_config.get("timeout", 30)
        self.request_interval = api_config.get("request_interval", 1.0)
        
        # 初始化 API 客户端
        self._client: Optional[APIClient] = None
    
    async def _get_client(self) -> APIClient:
        """获取 API 客户端"""
        if self._client is None:
            self._client = APIClient(
                base_url=self.BASE_URL,
                api_key=self.api_key,
                timeout=self.timeout,
                request_interval=self.request_interval,
            )
        return self._client
    
    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """执行 Semantic Scholar API 操作"""
        # 默认获取论文列表
        author_id = kwargs.get("author_id")
        if author_id:
            papers = await self.fetch_papers_by_author(author_id)
            return {"papers": [p.to_dict() for p in papers]}
        return {"error": "No author_id provided"}
    
    async def fetch_author_info(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        获取学者信息
        
        Args:
            author_id: Semantic Scholar Author ID
            
        Returns:
            学者信息字典
        """
        client = await self._get_client()
        
        try:
            endpoint = f"author/{author_id}"
            params = {"fields": ",".join(self.AUTHOR_FIELDS)}
            
            data = await client.get(endpoint, params)
            
            if data:
                logger.info(f"Fetched author info: {data.get('name', author_id)}")
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to fetch author {author_id}: {e}")
            return None
    
    async def fetch_papers_by_author(
        self,
        author_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[PaperMeta]:
        """
        获取学者的论文列表
        
        Args:
            author_id: Semantic Scholar Author ID
            limit: 返回论文数量上限
            offset: 偏移量（用于分页）
            
        Returns:
            论文元数据列表
        """
        client = await self._get_client()
        
        try:
            endpoint = f"author/{author_id}/papers"
            params = {
                "fields": ",".join(self.PAPER_FIELDS),
                "limit": min(limit, 100),  # API 最大限制 100
                "offset": offset,
            }
            
            data = await client.get(endpoint, params)
            
            if not data or "data" not in data:
                logger.warning(f"No papers found for author {author_id}")
                return []
            
            papers = []
            for paper_data in data["data"]:
                try:
                    paper = PaperMeta.from_semantic_scholar(paper_data)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper: {e}")
                    continue
            
            logger.info(f"Fetched {len(papers)} papers for author {author_id}")
            return papers
        except Exception as e:
            logger.error(f"Failed to fetch papers for author {author_id}: {e}")
            return []
    
    async def fetch_paper_details(self, paper_id: str) -> Optional[PaperMeta]:
        """
        获取论文详情
        
        Args:
            paper_id: Semantic Scholar Paper ID
            
        Returns:
            论文元数据
        """
        client = await self._get_client()
        
        try:
            endpoint = f"paper/{paper_id}"
            params = {"fields": ",".join(self.PAPER_FIELDS)}
            
            data = await client.get(endpoint, params)
            
            if data:
                paper = PaperMeta.from_semantic_scholar(data)
                logger.info(f"Fetched paper details: {paper.title[:50]}...")
                return paper
            return None
        except Exception as e:
            logger.error(f"Failed to fetch paper {paper_id}: {e}")
            return None
    
    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[tuple] = None,
    ) -> List[PaperMeta]:
        """
        搜索论文
        
        Args:
            query: 搜索查询
            limit: 返回数量上限
            year_range: 年份范围 (start_year, end_year)
            
        Returns:
            论文元数据列表
        """
        client = await self._get_client()
        
        try:
            endpoint = "paper/search"
            params = {
                "query": query,
                "fields": ",".join(self.PAPER_FIELDS),
                "limit": min(limit, 100),
            }
            
            if year_range:
                params["year"] = f"{year_range[0]}-{year_range[1]}"
            
            data = await client.get(endpoint, params)
            
            if not data or "data" not in data:
                return []
            
            papers = []
            for paper_data in data["data"]:
                try:
                    paper = PaperMeta.from_semantic_scholar(paper_data)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper: {e}")
                    continue
            
            logger.info(f"Search returned {len(papers)} papers for query: {query}")
            return papers
        except Exception as e:
            logger.error(f"Failed to search papers: {e}")
            return []
    
    async def update_scholar_metrics(self, scholar: Scholar) -> Scholar:
        """
        更新学者的学术指标
        
        Args:
            scholar: 学者对象
            
        Returns:
            更新后的学者对象
        """
        scholar_id = scholar.semantic_scholar_id
        if not scholar_id:
            return scholar
        
        author_info = await self.fetch_author_info(scholar_id)
        
        if author_info:
            scholar.h_index = author_info.get("hIndex")
            scholar.citation_count = author_info.get("citationCount")
            scholar.paper_count = author_info.get("paperCount")
            
            if author_info.get("affiliations"):
                scholar.affiliations = author_info["affiliations"]
            if author_info.get("homepage"):
                scholar.homepage = author_info["homepage"]
        
        return scholar
    
    async def close(self):
        """关闭 API 客户端"""
        if self._client:
            await self._client.close()
            self._client = None
