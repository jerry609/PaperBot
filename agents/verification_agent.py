# securipaperbot/agents/verification_agent.py
"""
VerificationAgent: Verifies scientific claims by finding supporting/refuting evidence.
Inspired by CIBER (Wang et al., 2025) - Corroborating and Refuting Evidence Retrieval.

This agent:
1. Extracts key claims from a paper abstract.
2. Searches for evidence in related literature (via Semantic Scholar API).
3. Labels each claim as: Supported, Refuted, Controversial, or Unverified.

Uses Mixin pattern:
- SemanticScholarMixin: S2 API search
- TextParsingMixin: Text extraction utilities
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from .mixins import SemanticScholarMixin, TextParsingMixin


class VerificationAgent(BaseAgent, SemanticScholarMixin, TextParsingMixin):
    """
    Verifies scientific claims by retrieving corroborating or refuting evidence.
    Uses multi-perspective questioning to find balanced evidence.
    """

    VERIFICATION_SYSTEM_PROMPT = """You are a scientific fact-checker.
Your job is to verify claims by finding and analyzing evidence.
Be objective and consider evidence from multiple perspectives.
When uncertain, acknowledge the uncertainty rather than making unfounded claims."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.init_s2_client(config)  # Initialize S2 mixin

    def _validate_input(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Validate that title and abstract are provided."""
        if not kwargs.get("title") or not kwargs.get("abstract"):
            return {"status": "error", "error": "VerificationAgent requires 'title' and 'abstract' arguments."}
        return None

    async def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Core execution: verify claims from paper."""
        title = kwargs.get("title")
        abstract = kwargs.get("abstract")
        num_claims = kwargs.get("num_claims", 3)
        search_limit = kwargs.get("search_limit", 10)
        
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
        """Extract the top N verifiable claims from the abstract."""
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
        
        # Use mixin method for parsing
        claims = self.extract_numbered_items(response, max_items=num_claims)
        return claims if claims else ["No extractable claims found."]

    async def _gather_evidence(self, claim: str, paper_title: str, limit: int) -> Dict[str, Any]:
        """
        Search for supporting and refuting evidence using Semantic Scholar API.
        Uses multi-perspective queries (CIBER style).
        """
        # Generate search queries from multiple perspectives
        supporting_query = await self._generate_search_query(claim, "supporting")
        refuting_query = await self._generate_search_query(claim, "refuting")
        
        # Search Semantic Scholar (using mixin)
        supporting_papers = await self.search_semantic_scholar(supporting_query, limit)
        refuting_papers = await self.search_semantic_scholar(refuting_query, limit)
        
        return {
            "supporting_query": supporting_query,
            "refuting_query": refuting_query,
            "supporting_papers": supporting_papers,
            "refuting_papers": refuting_papers
        }

    async def _generate_search_query(self, claim: str, perspective: str) -> str:
        """Generate a search query to find evidence from a specific perspective."""
        prompt = f"""Generate a search query to find {perspective} evidence for this claim:

**Claim:** {claim}

Return ONLY in 5-7 keywords suitable for an academic search engine.
Do not include quotes or boolean operators."""

        response = await self.ask_claude(prompt, max_tokens=50)
        return response.strip()

    async def _evaluate_evidence(self, claim: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the gathered evidence and determine a verdict for the claim."""
        supporting = evidence.get("supporting_papers", [])
        refuting = evidence.get("refuting_papers", [])
        
        # Format evidence for LLM (using mixin)
        supporting_str = self.format_papers_for_context(supporting[:3])
        refuting_str = self.format_papers_for_context(refuting[:3])
        
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
        
        # Parse response using mixin methods
        return {
            "verdict": self.extract_verdict(response),
            "confidence": self.extract_confidence(response),
            "reasoning": self.extract_field(response, "Reasoning"),
            "key_evidence": self.extract_field(response, "Key Evidence"),
            "raw_response": response
        }

    def _compute_overall_assessment(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute an overall assessment of the paper's claims."""
        verdicts = [r.get("verdict", {}).get("verdict", "Unverified") for r in results]
        
        supported = sum(1 for v in verdicts if "Supported" in str(v))
        refuted = sum(1 for v in verdicts if "Refuted" in str(v))
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
