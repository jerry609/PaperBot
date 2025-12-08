# securipaperbot/agents/research_agent.py

from typing import Dict, List, Any, Optional
import re
from .base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """
    单篇论文增强 Agent：摘要总结、代码链接提取、关键贡献生成
    （会议抓取已拆分至 ConferenceResearchAgent）
    """

    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.s2_api_key = config.get('semantic_scholar_api_key') if config else None

    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """仅支持单篇论文分析"""
        if "paper_title" in kwargs:
            return await self._analyze_single_paper(
                title=kwargs.get("paper_title"),
                paper_id=kwargs.get("paper_id"),
                abstract=kwargs.get("abstract")
            )
        raise ValueError("Invalid arguments for ResearchAgent.process (expect paper_title/paper_id/abstract)")

    async def _analyze_single_paper(self, title: str, paper_id: str, abstract: Optional[str]) -> Dict[str, Any]:
        """分析单篇论文"""
        result = {
            "paper_id": paper_id,
            "title": title,
            "executive_summary": None,
            "github_url": None,
            "key_contributions": [],
            "literature_grounding": None
        }
        
        if abstract:
            # 1. 生成摘要总结
            result["executive_summary"] = await self.summarize_abstract(abstract)
            
            # 2. 尝试从摘要中提取 GitHub 链接 (简单的正则)
            github_pattern = r'https?://github\.com/[\w-]+/[\w-]+'
            links = re.findall(github_pattern, abstract)
            if links:
                result["github_url"] = links[0]
        
        # 3. 如果启用了 LLM，可以生成 key_contributions 和 literature_grounding
        if self.client and abstract:
             try:
                 # 3.1 生成关键贡献
                 prompt = f"Based on the abstract below, list 3 key contributions of the paper '{title}'.\n\nAbstract: {abstract}"
                 response = await self.ask_claude(prompt)
                 contributions = [line.strip('- *') for line in response.split('\n') if line.strip()]
                 result["key_contributions"] = contributions[:3]
                 
                 # 3.2 Literature Grounding (Novelty Check)
                 result["literature_grounding"] = await self._check_novelty(title, abstract, result["key_contributions"])
                 
             except Exception as e:
                 self.log_error(e, {"context": "generate_analysis"})
             
        return result

    async def _check_novelty(self, title: str, abstract: str, contributions: List[str]) -> Dict[str, Any]:
        """
        Check novelty by searching prior art on Semantic Scholar.
        """
        import aiohttp
        
        # 1. Generate search queries for prior art
        queries = await self._generate_prior_art_queries(title, contributions)
        
        # 2. Search Semantic Scholar
        prior_art = []
        for q in queries[:2]:  # Limit to top 2 queries
            prior_art.extend(await self._search_s2(q, limit=5))
            
        # Deduplicate
        unique_art = {p['title']: p for p in prior_art}.values()
        
        # 3. Compare with prior art
        grounding_report = await self._compare_with_prior_art(title, abstract, list(unique_art)[:5])
        
        return {
            "queries": queries,
            "found_dates": [p.get('year') for p in unique_art],
            "analysis": grounding_report
        }

    async def _generate_prior_art_queries(self, title: str, contributions: List[str]) -> List[str]:
        """Generate keywords to find prior art."""
        contrib_str = "\n".join(contributions)
        prompt = f"""Generate 2-3 search queries to find PRIOR ART (existing work) that is similar to this paper.
        
**Title:** {title}
**Key Contributions:**
{contrib_str}

Return ONLY 2-3 search queries (one per line). Focus on the core problem and method concepts.
Example:
"large language model agent framework"
"automated code generation benchmarks"
"""
        response = await self.ask_claude(prompt, max_tokens=100)
        return [line.strip().strip('"') for line in response.splitlines() if line.strip()]

    async def _search_s2(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Semantic Scholar."""
        import aiohttp
        papers = []
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else {}
                params = {
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,year,citationCount"
                }
                async with session.get(f"{self.S2_API_BASE}/paper/search", params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        papers = data.get("data", [])
                    else:
                        self.logger.warning(f"S2 search failed: {resp.status}")
        except Exception as e:
            self.logger.warning(f"S2 search error: {e}")
        return papers

    async def _compare_with_prior_art(self, title: str, abstract: str, prior_art: List[Dict[str, Any]]) -> str:
        """Compare paper with found prior art."""
        art_desc = ""
        for i, p in enumerate(prior_art, 1):
             art_desc += f"{i}. {p.get('title')} ({p.get('year')})\n   Abstract: {p.get('abstract', '')[:150]}...\n\n"
             
        prompt = f"""Assess the NOVELTY of this paper by comparing it to the retrieved prior art.

**Current Paper:** {title}
**Abstract:** {abstract}

**Retrieved Prior Art:**
{art_desc if art_desc else "No relevant prior art found."}

Task:
1. Does the current paper propose something significantly different from the prior art?
2. Is the "Novelty" confirmed, or does it look like an incremental improvement?
3. Mention any specific paper from the list that looks very similar.

Output a concise "Literature Grounding Report" (3-4 sentences)."""

        return await self.ask_claude(prompt, max_tokens=300)

    async def summarize_abstract(self, abstract: str) -> str:
        """使用Claude总结摘要"""
        prompt = f"Please summarize the following academic paper abstract in 2-3 sentences:\n\n{abstract}"
        return await self.ask_claude(prompt, system="You are a helpful research assistant.")
    async def _extract_github_links(self, pdf_path: str) -> List[str]:
        """从PDF中提取GitHub链接"""
        github_pattern = r'https?://github\.com/[\w-]+/[\w-]+'
        links = []

        try:
            # 使用pdfplumber提取文本
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() for page in pdf.pages)
                links = re.findall(github_pattern, text)

            return list(set(links))  # 去重

        except Exception as e:
            self.log_error(e, {'pdf_path': pdf_path})
            return []

    def validate_config(self) -> bool:
        """验证配置是否完整"""
        required_keys = ['acm_base_url', 'ieee_base_url', 'download_path']
        return all(key in self.config for key in required_keys)