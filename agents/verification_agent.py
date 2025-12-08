# securipaperbot/agents/verification_agent.py
"""
VerificationAgent: Verifies scientific claims by finding supporting/refuting evidence.
Inspired by CIBER (Wang et al., 2025) - Corroborating and Refuting Evidence Retrieval.

This agent:
1. Extracts key claims from a paper abstract.
2. Searches for evidence in related literature (via Semantic Scholar API).
3. Labels each claim as: Supported, Refuted, Controversial, or Unverified.
"""

from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from .base_agent import BaseAgent


class VerificationAgent(BaseAgent):
    """
    Verifies scientific claims by retrieving corroborating or refuting evidence.
    Uses multi-perspective questioning to find balanced evidence.
    """

    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
    
    VERIFICATION_SYSTEM_PROMPT = """You are a scientific fact-checker.
Your job is to verify claims by finding and analyzing evidence.
Be objective and consider evidence from multiple perspectives.
When uncertain, acknowledge the uncertainty rather than making unfounded claims."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.s2_api_key = config.get('semantic_scholar_api_key') if config else None

    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Verify claims from a paper.
        
        Args (via kwargs):
            title: Paper title
            abstract: Paper abstract
            num_claims: Number of claims to extract (default: 3)
            search_limit: Number of related papers to search (default: 10)
        
        Returns:
            Dictionary with extracted claims and verification results.
        """
        title = kwargs.get("title")
        abstract = kwargs.get("abstract")
        num_claims = kwargs.get("num_claims", 3)
        search_limit = kwargs.get("search_limit", 10)
        
        if not title or not abstract:
            raise ValueError("VerificationAgent requires 'title' and 'abstract' arguments.")
        
        # Step 1: Extract key claims
        claims = await self._extract_claims(title, abstract, num_claims)
        
        # Step 2: For each claim, gather evidence
        verification_results = []
        for claim in claims:
            evidence = await self._gather_evidence(claim, title, search_limit)
            verdict = await self._evaluate_evidence(claim, evidence)
            verification_results.append({
                "claim": claim,
                "evidence": evidence,
                "verdict": verdict
            })
        
        # Step 3: Overall assessment
        overall = self._compute_overall_assessment(verification_results)
        
        return {
            "paper_title": title,
            "claims": verification_results,
            "overall_assessment": overall
        }

    async def _extract_claims(self, title: str, abstract: str, num_claims: int) -> List[str]:
        """
        Extract the top N verifiable claims from the abstract.
        """
        prompt = f"""Extract the {num_claims} most important **verifiable claims** from this paper.

**Title:** {title}
**Abstract:** {abstract}

A verifiable claim is a specific, factual statement that could be checked against evidence.
Exclude vague statements like "we propose a new method" - focus on specific results or comparisons.

Return each claim on a new line, numbered 1 through {num_claims}.
Example format:
1. [Specific claim about performance]
2. [Specific claim about comparison]
3. [Specific claim about findings]"""

        response = await self.ask_claude(prompt, system=self.VERIFICATION_SYSTEM_PROMPT, max_tokens=500)
        
        # Parse numbered claims
        claims = []
        import re
        matches = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)', response, re.DOTALL)
        for match in matches[:num_claims]:
            claim = match.strip()
            if claim:
                claims.append(claim)
        
        return claims if claims else ["No extractable claims found."]

    async def _gather_evidence(self, claim: str, paper_title: str, limit: int) -> Dict[str, Any]:
        """
        Search for supporting and refuting evidence using Semantic Scholar API.
        Uses multi-perspective queries (CIBER style).
        """
        # Generate search queries from multiple perspectives
        supporting_query = await self._generate_search_query(claim, "supporting")
        refuting_query = await self._generate_search_query(claim, "refuting")
        
        # Search Semantic Scholar
        supporting_papers = await self._search_semantic_scholar(supporting_query, limit)
        refuting_papers = await self._search_semantic_scholar(refuting_query, limit)
        
        return {
            "supporting_query": supporting_query,
            "refuting_query": refuting_query,
            "supporting_papers": supporting_papers,
            "refuting_papers": refuting_papers
        }

    async def _generate_search_query(self, claim: str, perspective: str) -> str:
        """
        Generate a search query to find evidence from a specific perspective.
        """
        prompt = f"""Generate a search query to find {perspective} evidence for this claim:

**Claim:** {claim}

Return ONLY in 5-7 keywords suitable for an academic search engine.
Do not include quotes or boolean operators."""

        response = await self.ask_claude(prompt, max_tokens=50)
        return response.strip()

    async def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers related to the query.
        """
        papers = []
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else {}
                params = {
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,citationCount,year,authors"
                }
                
                async with session.get(
                    f"{self.S2_API_BASE}/paper/search",
                    params=params,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for paper in data.get("data", []):
                            papers.append({
                                "title": paper.get("title"),
                                "abstract": paper.get("abstract", "")[:500] if paper.get("abstract") else "",
                                "citation_count": paper.get("citationCount", 0),
                                "year": paper.get("year"),
                                "authors": [a.get("name") for a in paper.get("authors", [])[:3]]
                            })
                    else:
                        self.logger.warning(f"S2 API returned status {resp.status}")
        except Exception as e:
            self.log_error(e, {"context": "semantic_scholar_search", "query": query})
        
        return papers

    async def _evaluate_evidence(self, claim: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the gathered evidence and determine a verdict for the claim.
        """
        supporting = evidence.get("supporting_papers", [])
        refuting = evidence.get("refuting_papers", [])
        
        # Format evidence for LLM
        supporting_str = self._format_papers(supporting[:3])
        refuting_str = self._format_papers(refuting[:3])
        
        prompt = f"""Evaluate this scientific claim based on the evidence.

**Claim:** {claim}

**Potentially Supporting Papers:**
{supporting_str if supporting_str else "No supporting papers found."}

**Potentially Refuting Papers:**
{refuting_str if refuting_str else "No refuting papers found."}

Determine:
1. **Verdict**: One of: "Strongly Supported", "Weakly Supported", "Controversial", "Weakly Refuted", "Strongly Refuted", or "Unverified"
2. **Confidence**: High, Medium, or Low
3. **Reasoning**: 2-3 sentences explaining your verdict based on the evidence.
4. **Key Evidence**: Which specific paper(s) most influenced your verdict?

Be objective. If evidence is mixed or insufficient, say so."""

        response = await self.ask_claude(prompt, system=self.VERIFICATION_SYSTEM_PROMPT, max_tokens=600)
        
        # Parse response
        verdict = self._extract_verdict(response)
        
        return {
            "verdict": verdict,
            "confidence": self._extract_confidence(response),
            "reasoning": self._extract_field(response, "Reasoning"),
            "key_evidence": self._extract_field(response, "Key Evidence"),
            "raw_response": response
        }

    def _format_papers(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers for LLM context."""
        if not papers:
            return ""
        formatted = []
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p.get("authors", [])[:2])
            if len(p.get("authors", [])) > 2:
                authors += " et al."
            formatted.append(
                f"{i}. \"{p.get('title', 'Unknown')}\" ({p.get('year', 'N/A')})\n"
                f"   Authors: {authors}\n"
                f"   Citations: {p.get('citation_count', 0)}\n"
                f"   Abstract: {p.get('abstract', 'N/A')[:200]}..."
            )
        return "\n".join(formatted)

    def _extract_verdict(self, text: str) -> str:
        """Extract the verdict from LLM response."""
        import re
        verdicts = [
            "Strongly Supported", "Weakly Supported", "Controversial",
            "Weakly Refuted", "Strongly Refuted", "Unverified"
        ]
        for v in verdicts:
            if v.lower() in text.lower():
                return v
        return "Unverified"

    def _extract_confidence(self, text: str) -> str:
        """Extract confidence level from response."""
        import re
        for conf in ["High", "Medium", "Low"]:
            if conf.lower() in text.lower():
                return conf
        return "Medium"

    def _extract_field(self, text: str, field: str) -> str:
        """Extract a text field value."""
        import re
        pattern = rf'{field}[:\s]*(.+?)(?=\n\*\*|\n\d\.|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:500]
        return ""

    def _compute_overall_assessment(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute an overall assessment of the paper's claims.
        """
        verdicts = [r.get("verdict", {}).get("verdict", "Unverified") for r in results]
        
        supported = sum(1 for v in verdicts if "Supported" in v)
        refuted = sum(1 for v in verdicts if "Refuted" in v)
        controversial = sum(1 for v in verdicts if v == "Controversial")
        unverified = sum(1 for v in verdicts if v == "Unverified")
        
        total = len(verdicts) or 1
        
        if supported / total > 0.6:
            overall = "Claims are generally well-supported by literature."
        elif refuted / total > 0.4:
            overall = "Caution: Some claims may conflict with existing literature."
        elif controversial / total > 0.4:
            overall = "Paper addresses controversial topics with mixed evidence."
        else:
            overall = "Claims require further verification - limited evidence found."
        
        return {
            "summary": overall,
            "supported_count": supported,
            "refuted_count": refuted,
            "controversial_count": controversial,
            "unverified_count": unverified
        }
