# src/paperbot/domain/influence/analyzers/citation_context.py
"""
Citation Context Analyzer.

Analyzes the sentiment of citations to determine:
- Positive citations (extensions, validations)
- Negative citations (critiques, refutations)
- Neutral citations (simple references)
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..result import (
    CitationSentiment, CitationContext, CitationSentimentResult
)

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


class CitationContextAnalyzer:
    """
    Analyze citation contexts to determine sentiment.
    
    Uses LLM to classify each citation as positive/negative/neutral
    based on the surrounding text context.
    """
    
    SENTIMENT_PROMPT = """Analyze this citation context and classify the sentiment.

Cited Paper: {cited_title}
Citing Paper: {citing_title}
Citation Context: "{context}"

Classify the citation as:
- POSITIVE: The citing paper extends, validates, builds upon, or praises the cited work
- NEGATIVE: The citing paper critiques, refutes, identifies limitations, or disagrees
- NEUTRAL: Simple reference without strong opinion, background mention

Output JSON only:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "reason": "Brief explanation (1 sentence)"
}}
"""

    BATCH_SENTIMENT_PROMPT = """Analyze these citation contexts for the paper "{cited_title}".

Citations:
{citations_text}

For each citation, classify as POSITIVE (extends/validates), NEGATIVE (critiques/refutes), or NEUTRAL (simple reference).

Output JSON array:
[
    {{"index": 0, "sentiment": "positive|negative|neutral", "confidence": 0.8, "reason": "..."}},
    ...
]
"""

    def __init__(self, max_contexts: int = 20):
        """
        Initialize the analyzer.
        
        Args:
            max_contexts: Maximum number of contexts to analyze per paper
        """
        self.max_contexts = max_contexts
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_single(
        self,
        cited_title: str,
        citing_title: str,
        context_text: str,
        citing_paper_id: str = "",
    ) -> CitationContext:
        """
        Analyze a single citation context.
        
        Args:
            cited_title: Title of the paper being cited
            citing_title: Title of the paper doing the citing
            context_text: The text surrounding the citation
            citing_paper_id: ID of the citing paper
            
        Returns:
            CitationContext with sentiment classification
        """
        result = CitationContext(
            citing_paper_id=citing_paper_id,
            citing_paper_title=citing_title,
            context_text=context_text,
            sentiment=CitationSentiment.NEUTRAL,
            confidence=0.5,
        )
        
        if not query or not ClaudeAgentOptions:
            return result
        
        try:
            prompt = self.SENTIMENT_PROMPT.format(
                cited_title=cited_title,
                citing_title=citing_title,
                context=context_text[:500],
            )
            
            llm_result = query(
                prompt=prompt,
                options=ClaudeAgentOptions(max_tokens=200)
            )
            
            parsed = self._parse_single_response(llm_result.response)
            if parsed:
                result.sentiment = CitationSentiment(parsed.get("sentiment", "neutral"))
                result.confidence = parsed.get("confidence", 0.5)
                result.reason = parsed.get("reason", "")
                
        except Exception as e:
            self.logger.warning(f"Single citation analysis failed: {e}")
        
        return result
    
    def analyze_batch(
        self,
        cited_title: str,
        citations: List[Dict[str, Any]],
    ) -> CitationSentimentResult:
        """
        Analyze multiple citation contexts in batch.
        
        Args:
            cited_title: Title of the paper being assessed
            citations: List of dicts with keys: citing_title, context, citing_id
            
        Returns:
            CitationSentimentResult with aggregated analysis
        """
        # Limit to max contexts
        citations = citations[:self.max_contexts]
        
        result = CitationSentimentResult(
            total_analyzed=len(citations),
        )
        
        if not citations:
            return result
        
        # Try batch analysis first
        if query and ClaudeAgentOptions and len(citations) > 1:
            batch_result = self._analyze_batch_llm(cited_title, citations)
            if batch_result:
                return batch_result
        
        # Fallback to individual analysis
        contexts = []
        for cit in citations:
            ctx = self.analyze_single(
                cited_title=cited_title,
                citing_title=cit.get("citing_title", "Unknown"),
                context_text=cit.get("context", ""),
                citing_paper_id=cit.get("citing_id", ""),
            )
            contexts.append(ctx)
        
        return self._aggregate_results(contexts)
    
    def _analyze_batch_llm(
        self, cited_title: str, citations: List[Dict[str, Any]]
    ) -> Optional[CitationSentimentResult]:
        """Use LLM for batch analysis."""
        try:
            # Format citations for prompt
            citations_text = "\n".join([
                f"[{i}] From \"{c.get('citing_title', 'Unknown')}\": \"{c.get('context', '')[:200]}...\""
                for i, c in enumerate(citations)
            ])
            
            prompt = self.BATCH_SENTIMENT_PROMPT.format(
                cited_title=cited_title,
                citations_text=citations_text,
            )
            
            llm_result = query(
                prompt=prompt,
                options=ClaudeAgentOptions(max_tokens=1500)
            )
            
            parsed = self._parse_batch_response(llm_result.response)
            if not parsed:
                return None
            
            # Build contexts from parsed results
            contexts = []
            for item in parsed:
                idx = item.get("index", 0)
                if idx < len(citations):
                    cit = citations[idx]
                    ctx = CitationContext(
                        citing_paper_id=cit.get("citing_id", ""),
                        citing_paper_title=cit.get("citing_title", "Unknown"),
                        context_text=cit.get("context", ""),
                        sentiment=CitationSentiment(item.get("sentiment", "neutral")),
                        confidence=item.get("confidence", 0.5),
                        reason=item.get("reason", ""),
                    )
                    contexts.append(ctx)
            
            return self._aggregate_results(contexts)
            
        except Exception as e:
            self.logger.warning(f"Batch LLM analysis failed: {e}")
            return None
    
    def _aggregate_results(self, contexts: List[CitationContext]) -> CitationSentimentResult:
        """Aggregate individual contexts into summary result."""
        result = CitationSentimentResult(
            total_analyzed=len(contexts),
            contexts=contexts,
        )
        
        for ctx in contexts:
            if ctx.sentiment == CitationSentiment.POSITIVE:
                result.positive_count += 1
            elif ctx.sentiment == CitationSentiment.NEGATIVE:
                result.negative_count += 1
                # Collect notable critiques
                if ctx.reason:
                    result.notable_critiques.append(
                        f"{ctx.citing_paper_title}: {ctx.reason}"
                    )
            else:
                result.neutral_count += 1
        
        # Compute sentiment score
        # Score = 50 + (positive_ratio - negative_ratio) * 50
        if result.total_analyzed > 0:
            diff = result.positive_ratio - result.negative_ratio
            result.sentiment_score = max(0, min(100, 50 + diff * 50))
        
        return result
    
    def _parse_single_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse single JSON response."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return None
    
    def _parse_batch_response(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """Parse batch JSON array response."""
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return None
    
    def compute_sentiment_score(self, result: CitationSentimentResult) -> float:
        """
        Convert sentiment result to a score adjustment.
        
        Returns:
            Adjustment value (-10 to +10) to apply to influence score
        """
        # High positive ratio: +10 bonus
        # High negative ratio: -10 penalty
        # Neutral: 0
        if result.total_analyzed == 0:
            return 0.0
        
        diff = result.positive_ratio - result.negative_ratio
        return diff * 10  # -10 to +10 range
