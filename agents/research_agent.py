# securipaperbot/agents/research_agent.py

from typing import Dict, List, Any, Optional
import re
from .base_agent import BaseAgent
from .mixins import SemanticScholarMixin, TextParsingMixin


class ResearchAgent(BaseAgent, SemanticScholarMixin, TextParsingMixin):
    """
    单篇论文增强 Agent：摘要总结、代码链接提取、关键贡献生成、文献背景分析
    （会议抓取已拆分至 ConferenceResearchAgent）
    
    使用 Mixin 模式：
    - SemanticScholarMixin: S2 API 搜索
    - TextParsingMixin: 文本解析工具
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.init_s2_client(config)  # Initialize S2 mixin

    def _validate_input(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Validate that paper_title is provided."""
        if "paper_title" not in kwargs:
            return {"status": "error", "error": "Missing required argument: paper_title"}
        return None

    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Core execution: analyze single paper."""
        return await self._analyze_single_paper(
            title=kwargs.get("paper_title"),
            paper_id=kwargs.get("paper_id"),
            abstract=kwargs.get("abstract")
        )

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
                contributions = self.extract_numbered_items(response, max_items=3)
                if not contributions:
                    contributions = [line.strip('- *') for line in response.split('\n') if line.strip()][:3]
                result["key_contributions"] = contributions
                
                # 3.2 Literature Grounding (Novelty Check)
                result["literature_grounding"] = await self._check_novelty(title, abstract, result["key_contributions"])
                
            except Exception as e:
                self.log_error(e, {"context": "generate_analysis"})
             
        return result

    async def _check_novelty(self, title: str, abstract: str, contributions: List[str]) -> Dict[str, Any]:
        """Check novelty by searching prior art on Semantic Scholar."""
        # 1. Generate search queries for prior art
        queries = await self._generate_prior_art_queries(title, contributions)
        
        # 2. Search Semantic Scholar (using mixin)
        prior_art = []
        for q in queries[:2]:
            papers = await self.search_semantic_scholar(q, limit=5)
            prior_art.extend(papers)
            
        # Deduplicate
        unique_art = list({p.get('title', ''): p for p in prior_art if p.get('title')}.values())
        
        # 3. Compare with prior art
        grounding_report = await self._compare_with_prior_art(title, abstract, unique_art[:5])
        
        return {
            "queries": queries,
            "found_dates": [p.get('year') for p in unique_art],
            "analysis": grounding_report
        }

    async def _generate_prior_art_queries(self, title: str, contributions: List[str]) -> List[str]:
        """Generate keywords to find prior art."""
        contrib_str = "\n".join(contributions)
        prompt = f"""Generate 2-3 search queries to find PRIOR ART (existing work) similar to this paper.
        
**Title:** {title}
**Key Contributions:**
{contrib_str}

Return ONLY 2-3 search queries (one per line). Focus on core problem and method concepts.
Example:
"large language model agent framework"
"automated code generation benchmarks"
"""
        response = await self.ask_claude(prompt, max_tokens=100)
        return [line.strip().strip('"') for line in response.splitlines() if line.strip()]

    async def _compare_with_prior_art(self, title: str, abstract: str, prior_art: List[Dict[str, Any]]) -> str:
        """Compare paper with found prior art."""
        # Use mixin to format papers
        art_desc = self.format_papers_for_context(prior_art, max_papers=5)
             
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
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() for page in pdf.pages)
                links = re.findall(github_pattern, text)
            return list(set(links))
        except Exception as e:
            self.log_error(e, {'pdf_path': pdf_path})
            return []

    def validate_config(self) -> bool:
        """验证配置是否完整"""
        required_keys = ['acm_base_url', 'ieee_base_url', 'download_path']
        return all(key in self.config for key in required_keys)