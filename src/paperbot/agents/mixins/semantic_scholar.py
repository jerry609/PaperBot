# src/paperbot/agents/mixins/semantic_scholar.py
"""
Semantic Scholar API Mixin for PaperBot Agents.
Provides shared S2 search functionality to avoid code duplication.
"""

from typing import Dict, List, Any, Optional
import logging

from paperbot.infrastructure.api_clients.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class SemanticScholarMixin:
    """
    Mixin providing Semantic Scholar API integration.
    
    Usage:
        class MyAgent(BaseAgent, SemanticScholarMixin):
            def __init__(self, config):
                super().__init__(config)
                self.init_s2_client(config)
    """
    
    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

    def init_s2_client(self, config: Optional[Dict[str, Any]] = None):
        """Initialize S2 client with API key from config."""
        self._s2_api_key = config.get('semantic_scholar_api_key') if config else None
        self._s2_client: Optional[SemanticScholarClient] = None

    def _get_s2_client(self) -> SemanticScholarClient:
        if self._s2_client is None:
            request_interval = 0.3 if self._s2_api_key else 3.0
            self._s2_client = SemanticScholarClient(
                api_key=self._s2_api_key,
                request_interval=request_interval,
            )
        return self._s2_client
    
    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            fields: List of fields to retrieve (default: title, abstract, year, citationCount)
        
        Returns:
            List of paper dictionaries
        """
        if fields is None:
            fields = ["title", "abstract", "year", "citationCount", "authors"]

        try:
            client = self._get_s2_client()
            return await client.search_papers(query=query, limit=limit, fields=fields)
        except Exception as e:
            logger.warning(f"S2 search error: {e}")
            return []
    
    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID or DOI
        
        Returns:
            Paper details dictionary or None if not found
        """
        try:
            client = self._get_s2_client()
            return await client.get_paper(
                paper_id,
                fields=["title", "abstract", "year", "citationCount", "authors", "references", "citations"],
            )
        except Exception as e:
            logger.warning(f"S2 paper lookup error: {e}")

        return None
    
    def format_papers_for_context(self, papers: List[Dict[str, Any]], max_papers: int = 5) -> str:
        """
        Format papers for LLM context.
        
        Args:
            papers: List of paper dictionaries
            max_papers: Maximum number of papers to include
        
        Returns:
            Formatted string for LLM prompt
        """
        if not papers:
            return ""
        
        formatted = []
        for i, p in enumerate(papers[:max_papers], 1):
            authors = p.get("authors", [])
            if isinstance(authors, list) and authors:
                if isinstance(authors[0], dict):
                    author_names = [a.get("name", "") for a in authors[:2]]
                else:
                    author_names = authors[:2]
                author_str = ", ".join(author_names)
                if len(authors) > 2:
                    author_str += " et al."
            else:
                author_str = "Unknown"
            
            abstract = p.get("abstract", "")
            if abstract and len(abstract) > 200:
                abstract = abstract[:200] + "..."
            
            formatted.append(
                f"{i}. \"{p.get('title', 'Unknown')}\" ({p.get('year', 'N/A')})\n"
                f"   Authors: {author_str}\n"
                f"   Citations: {p.get('citationCount', 0)}\n"
                f"   Abstract: {abstract or 'N/A'}"
            )
        
        return "\n".join(formatted)

    async def close_s2_client(self) -> None:
        if self._s2_client is not None:
            await self._s2_client.close()
            self._s2_client = None
