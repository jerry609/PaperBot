"""
学术搜索工具集

参考: BettaFish/QueryEngine/tools/search.py
适配: PaperBot 学者追踪 - Semantic Scholar API

专为 AI Agent 设计的学术搜索工具集，提供:
- 学者搜索
- 论文搜索
- 引用关系搜索
- 作者论文搜索
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    import requests
except ImportError:
    raise ImportError("requests 库未安装，请运行 `pip install requests` 进行安装。")

from paperbot.utils.retry_helper import with_graceful_retry, SEMANTIC_SCHOLAR_RETRY_CONFIG


# ===== 数据结构定义 =====

@dataclass
class AuthorResult:
    """作者搜索结果"""
    author_id: str
    name: str
    affiliations: List[str] = field(default_factory=list)
    paper_count: int = 0
    citation_count: int = 0
    h_index: int = 0
    url: Optional[str] = None


@dataclass
class PaperResult:
    """论文搜索结果"""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: int = 0
    authors: List[Dict[str, str]] = field(default_factory=list)
    url: Optional[str] = None
    open_access_pdf: Optional[str] = None
    fields_of_study: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None


@dataclass
class SearchResponse:
    """搜索响应封装"""
    query: str
    total: int = 0
    offset: int = 0
    authors: List[AuthorResult] = field(default_factory=list)
    papers: List[PaperResult] = field(default_factory=list)
    response_time: Optional[float] = None


# ===== Semantic Scholar 搜索客户端 =====

class SemanticScholarSearch:
    """
    Semantic Scholar 学术搜索工具集
    
    每个公共方法都设计为供 AI Agent 独立调用的工具
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # API 字段定义
    AUTHOR_FIELDS = "authorId,name,affiliations,paperCount,citationCount,hIndex,url"
    PAPER_FIELDS = "paperId,title,abstract,year,venue,citationCount,authors,url,openAccessPdf,fieldsOfStudy,publicationDate"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客户端
        
        Args:
            api_key: Semantic Scholar API Key (可选，有 key 可提高限流)
        """
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self._headers = {"Accept": "application/json"}
        if self.api_key:
            self._headers["x-api-key"] = self.api_key
    
    def _parse_author(self, data: Dict[str, Any]) -> AuthorResult:
        """解析作者数据"""
        affiliations = data.get("affiliations") or []
        if isinstance(affiliations, list):
            affiliations = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in affiliations]
        
        return AuthorResult(
            author_id=data.get("authorId", ""),
            name=data.get("name", ""),
            affiliations=affiliations,
            paper_count=data.get("paperCount", 0),
            citation_count=data.get("citationCount", 0),
            h_index=data.get("hIndex", 0),
            url=data.get("url"),
        )
    
    def _parse_paper(self, data: Dict[str, Any]) -> PaperResult:
        """解析论文数据"""
        authors = []
        for a in data.get("authors") or []:
            authors.append({
                "authorId": a.get("authorId", ""),
                "name": a.get("name", ""),
            })
        
        open_access_pdf = None
        pdf_info = data.get("openAccessPdf")
        if pdf_info and isinstance(pdf_info, dict):
            open_access_pdf = pdf_info.get("url")
        
        return PaperResult(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citationCount", 0),
            authors=authors,
            url=data.get("url"),
            open_access_pdf=open_access_pdf,
            fields_of_study=data.get("fieldsOfStudy") or [],
            publication_date=data.get("publicationDate"),
        )
    
    @with_graceful_retry(SEMANTIC_SCHOLAR_RETRY_CONFIG, default_return=SearchResponse(query="搜索失败"))
    def search_authors(self, query: str, limit: int = 10) -> SearchResponse:
        """
        【工具】搜索学者: 根据姓名搜索学者信息
        
        Args:
            query: 学者姓名或关键词
            limit: 返回结果数量上限
            
        Returns:
            包含学者列表的搜索响应
        """
        logger.info(f"--- TOOL: 搜索学者 (query: {query}) ---")
        
        start_time = datetime.now()
        
        url = f"{self.BASE_URL}/author/search"
        params = {
            "query": query,
            "fields": self.AUTHOR_FIELDS,
            "limit": min(limit, 100),
        }
        
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            authors = [self._parse_author(a) for a in data.get("data", [])]
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return SearchResponse(
                query=query,
                total=data.get("total", len(authors)),
                authors=authors,
                response_time=elapsed,
            )
        except Exception as e:
            logger.error(f"搜索学者失败: {e}")
            raise
    
    @with_graceful_retry(SEMANTIC_SCHOLAR_RETRY_CONFIG, default_return=SearchResponse(query="搜索失败"))
    def search_papers(self, query: str, limit: int = 20, year_range: Optional[str] = None) -> SearchResponse:
        """
        【工具】搜索论文: 根据关键词搜索论文
        
        Args:
            query: 搜索关键词
            limit: 返回结果数量上限
            year_range: 年份范围，格式如 "2020-2024" 或 "2023-"
            
        Returns:
            包含论文列表的搜索响应
        """
        logger.info(f"--- TOOL: 搜索论文 (query: {query}) ---")
        
        start_time = datetime.now()
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "fields": self.PAPER_FIELDS,
            "limit": min(limit, 100),
        }
        
        if year_range:
            params["year"] = year_range
        
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = [self._parse_paper(p) for p in data.get("data", [])]
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return SearchResponse(
                query=query,
                total=data.get("total", len(papers)),
                offset=data.get("offset", 0),
                papers=papers,
                response_time=elapsed,
            )
        except Exception as e:
            logger.error(f"搜索论文失败: {e}")
            raise
    
    @with_graceful_retry(SEMANTIC_SCHOLAR_RETRY_CONFIG, default_return=None)
    def get_author(self, author_id: str) -> Optional[AuthorResult]:
        """
        【工具】获取学者详情: 根据 ID 获取学者详细信息
        
        Args:
            author_id: Semantic Scholar 作者 ID
            
        Returns:
            学者详细信息
        """
        logger.info(f"--- TOOL: 获取学者详情 (id: {author_id}) ---")
        
        url = f"{self.BASE_URL}/author/{author_id}"
        params = {"fields": self.AUTHOR_FIELDS}
        
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_author(data)
        except Exception as e:
            logger.error(f"获取学者详情失败: {e}")
            return None
    
    @with_graceful_retry(SEMANTIC_SCHOLAR_RETRY_CONFIG, default_return=[])
    def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PaperResult]:
        """
        【工具】获取学者论文: 获取指定学者的论文列表
        
        Args:
            author_id: Semantic Scholar 作者 ID
            limit: 返回结果数量上限
            offset: 分页偏移量
            
        Returns:
            论文列表
        """
        logger.info(f"--- TOOL: 获取学者论文 (id: {author_id}) ---")
        
        url = f"{self.BASE_URL}/author/{author_id}/papers"
        params = {
            "fields": self.PAPER_FIELDS,
            "limit": min(limit, 1000),
            "offset": offset,
        }
        
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return [self._parse_paper(p) for p in data.get("data", [])]
        except Exception as e:
            logger.error(f"获取学者论文失败: {e}")
            return []
    
    @with_graceful_retry(SEMANTIC_SCHOLAR_RETRY_CONFIG, default_return=None)
    def get_paper(self, paper_id: str) -> Optional[PaperResult]:
        """
        【工具】获取论文详情: 根据 ID 获取论文详细信息
        
        Args:
            paper_id: Semantic Scholar 论文 ID
            
        Returns:
            论文详细信息
        """
        logger.info(f"--- TOOL: 获取论文详情 (id: {paper_id}) ---")
        
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": self.PAPER_FIELDS}
        
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_paper(data)
        except Exception as e:
            logger.error(f"获取论文详情失败: {e}")
            return None
    
    @with_graceful_retry(SEMANTIC_SCHOLAR_RETRY_CONFIG, default_return=[])
    def get_paper_citations(self, paper_id: str, limit: int = 100) -> List[PaperResult]:
        """
        【工具】获取引用论文: 获取引用了指定论文的论文列表
        
        Args:
            paper_id: Semantic Scholar 论文 ID
            limit: 返回结果数量上限
            
        Returns:
            引用论文列表
        """
        logger.info(f"--- TOOL: 获取引用论文 (id: {paper_id}) ---")
        
        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        params = {
            "fields": "citingPaper." + self.PAPER_FIELDS,
            "limit": min(limit, 1000),
        }
        
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get("data", []):
                citing_paper = item.get("citingPaper", {})
                if citing_paper:
                    papers.append(self._parse_paper(citing_paper))
            return papers
        except Exception as e:
            logger.error(f"获取引用论文失败: {e}")
            return []
    
    @with_graceful_retry(SEMANTIC_SCHOLAR_RETRY_CONFIG, default_return=[])
    def get_paper_references(self, paper_id: str, limit: int = 100) -> List[PaperResult]:
        """
        【工具】获取参考文献: 获取指定论文引用的论文列表
        
        Args:
            paper_id: Semantic Scholar 论文 ID
            limit: 返回结果数量上限
            
        Returns:
            参考文献列表
        """
        logger.info(f"--- TOOL: 获取参考文献 (id: {paper_id}) ---")
        
        url = f"{self.BASE_URL}/paper/{paper_id}/references"
        params = {
            "fields": "citedPaper." + self.PAPER_FIELDS,
            "limit": min(limit, 1000),
        }
        
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            for item in data.get("data", []):
                cited_paper = item.get("citedPaper", {})
                if cited_paper:
                    papers.append(self._parse_paper(cited_paper))
            return papers
        except Exception as e:
            logger.error(f"获取参考文献失败: {e}")
            return []
    
    def search_security_papers(
        self,
        query: str,
        venues: Optional[List[str]] = None,
        limit: int = 20,
    ) -> SearchResponse:
        """
        【工具】搜索安全论文: 专门搜索安全领域顶会论文
        
        Args:
            query: 搜索关键词
            venues: 限定的会议/期刊列表，默认为四大安全顶会
            limit: 返回结果数量上限
            
        Returns:
            包含论文列表的搜索响应
        """
        if venues is None:
            venues = [
                "IEEE S&P",
                "IEEE Symposium on Security and Privacy",
                "CCS",
                "ACM Conference on Computer and Communications Security",
                "USENIX Security",
                "NDSS",
                "Network and Distributed System Security",
            ]
        
        # 构建包含会议名称的查询
        venue_query = " OR ".join([f'venue:"{v}"' for v in venues])
        enhanced_query = f"({query}) AND ({venue_query})"
        
        logger.info(f"--- TOOL: 搜索安全论文 (query: {query}) ---")
        
        return self.search_papers(enhanced_query, limit=limit)


# ===== 结果排序器 =====

class SearchResultRanker:
    """
    搜索结果排序器
    
    提供多种排序策略用于对搜索结果进行排序
    """
    
    @staticmethod
    def rank_papers_by_citation(papers: List[PaperResult], descending: bool = True) -> List[PaperResult]:
        """按引用数排序"""
        return sorted(papers, key=lambda p: p.citation_count, reverse=descending)
    
    @staticmethod
    def rank_papers_by_year(papers: List[PaperResult], descending: bool = True) -> List[PaperResult]:
        """按年份排序"""
        return sorted(papers, key=lambda p: p.year or 0, reverse=descending)
    
    @staticmethod
    def rank_papers_by_relevance(
        papers: List[PaperResult],
        query: str,
        citation_weight: float = 0.3,
        recency_weight: float = 0.3,
        title_match_weight: float = 0.4,
    ) -> List[PaperResult]:
        """
        综合相关性排序
        
        Args:
            papers: 论文列表
            query: 原始查询
            citation_weight: 引用数权重
            recency_weight: 时效性权重
            title_match_weight: 标题匹配权重
        """
        current_year = datetime.now().year
        query_lower = query.lower()
        
        def score(paper: PaperResult) -> float:
            # 引用分数 (归一化到 0-1)
            max_citations = max(p.citation_count for p in papers) if papers else 1
            citation_score = paper.citation_count / max_citations if max_citations > 0 else 0
            
            # 时效分数 (最近5年为满分)
            year = paper.year or 2000
            recency_score = max(0, min(1, (year - current_year + 5) / 5))
            
            # 标题匹配分数
            title_lower = (paper.title or "").lower()
            title_score = sum(1 for word in query_lower.split() if word in title_lower) / len(query_lower.split())
            
            return (
                citation_weight * citation_score +
                recency_weight * recency_score +
                title_match_weight * title_score
            )
        
        return sorted(papers, key=score, reverse=True)
    
    @staticmethod
    def rank_authors_by_influence(
        authors: List[AuthorResult],
        h_index_weight: float = 0.4,
        citation_weight: float = 0.3,
        paper_weight: float = 0.3,
    ) -> List[AuthorResult]:
        """
        按影响力排序学者
        
        Args:
            authors: 学者列表
            h_index_weight: H-index 权重
            citation_weight: 引用数权重
            paper_weight: 论文数权重
        """
        if not authors:
            return authors
        
        max_h = max(a.h_index for a in authors) or 1
        max_citations = max(a.citation_count for a in authors) or 1
        max_papers = max(a.paper_count for a in authors) or 1
        
        def score(author: AuthorResult) -> float:
            return (
                h_index_weight * (author.h_index / max_h) +
                citation_weight * (author.citation_count / max_citations) +
                paper_weight * (author.paper_count / max_papers)
            )
        
        return sorted(authors, key=score, reverse=True)


__all__ = [
    # 数据类
    "AuthorResult",
    "PaperResult",
    "SearchResponse",
    # 搜索客户端
    "SemanticScholarSearch",
    # 排序器
    "SearchResultRanker",
]
