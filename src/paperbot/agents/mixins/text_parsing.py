# src/paperbot/agents/mixins/text_parsing.py
"""
Text Parsing Mixin for PaperBot Agents.
Provides shared text extraction utilities to avoid code duplication.
"""

import re
from typing import List, Optional


class TextParsingMixin:
    """
    Mixin providing text parsing utilities for LLM responses.
    
    Usage:
        class MyAgent(BaseAgent, TextParsingMixin):
            pass
    """
    
    def extract_field(self, text: str, field: str, max_length: int = 500) -> str:
        """
        Extract a labeled field value from LLM response.
        
        Args:
            text: Full LLM response text
            field: Field name to extract (e.g., "Reasoning", "Summary")
            max_length: Maximum length of extracted content
        
        Returns:
            Extracted field value or empty string
        """
        # Pattern: **Field:** value or Field: value
        pattern = rf'{field}[:\s]*(.+?)(?=\n\*\*|\n\d\.|\n#|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:max_length]
        return ""
    
    def extract_score(self, text: str, metric: str, default: int = 5) -> int:
        """
        Extract a numeric score (1-10) for a given metric.
        
        Args:
            text: Full LLM response text
            metric: Metric name to look for
            default: Default score if not found
        
        Returns:
            Extracted score or default
        """
        # Pattern: metric: N or metric: N/10
        pattern = rf'{metric}[:\s]*(\d+)(?:/\d+)?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(1, min(10, score))
        return default
    
    def extract_bullet_points(self, text: str, section: str) -> List[str]:
        """
        Extract bullet points from a section of text.
        
        Args:
            text: Full LLM response text
            section: Section name to extract bullets from
        
        Returns:
            List of bullet point strings
        """
        # Find section
        pattern = rf'{section}[:\s]*\n((?:[-*•]\s*.+\n?)+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            bullets_text = match.group(1)
            bullets = re.findall(r'[-*•]\s*(.+?)(?=\n[-*•]|\Z)', bullets_text, re.DOTALL)
            return [b.strip() for b in bullets if b.strip()]
        return []
    
    def extract_verdict(self, text: str, verdicts: Optional[List[str]] = None) -> str:
        """
        Extract a verdict label from LLM response.
        
        Args:
            text: Full LLM response text
            verdicts: List of valid verdict labels (default: review verdicts)
        
        Returns:
            Matched verdict or "Unknown"
        """
        if verdicts is None:
            verdicts = [
                "Strongly Supported", "Weakly Supported", "Controversial",
                "Weakly Refuted", "Strongly Refuted", "Unverified",
                "Accept", "Reject", "Borderline"
            ]
        
        for v in verdicts:
            if v.lower() in text.lower():
                return v
        return "Unknown"
    
    def extract_confidence(self, text: str) -> str:
        """
        Extract confidence level from response.
        
        Args:
            text: Full LLM response text
        
        Returns:
            Confidence level (High/Medium/Low)
        """
        for conf in ["High", "Medium", "Low"]:
            if conf.lower() in text.lower():
                return conf
        return "Medium"
    
    def extract_numbered_items(self, text: str, max_items: int = 10) -> List[str]:
        """
        Extract numbered items (1. xxx, 2. xxx) from text.
        
        Args:
            text: Full LLM response text
            max_items: Maximum number of items to extract
        
        Returns:
            List of extracted items
        """
        pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches[:max_items] if m.strip()]
    
    def clean_code_block(self, text: str) -> str:
        """
        Remove markdown code block markers from text.
        
        Args:
            text: Text potentially containing ```python ... ```
        
        Returns:
            Cleaned code content
        """
        # Remove ```python or ``` markers
        pattern = r'```(?:\w+)?\n?(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
