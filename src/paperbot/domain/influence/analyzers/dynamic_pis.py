# src/paperbot/domain/influence/analyzers/dynamic_pis.py
"""
Dynamic PIS (PaperBot Impact Score) Calculator.

Computes dynamic influence indicators:
- Citation velocity (recent citation growth)
- Momentum score (trend classification)
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from ..result import CitationVelocity, CitationTrend

logger = logging.getLogger(__name__)


class DynamicPISCalculator:
    """
    Calculate dynamic influence scores based on citation velocity.
    
    Key metrics:
    - Recent citations (default: last 6 months)
    - Growth rate compared to historical average
    - Trend classification (accelerating/stable/declining)
    """
    
    # Thresholds for trend classification
    ACCELERATING_THRESHOLD = 0.2   # 20% above average = accelerating
    DECLINING_THRESHOLD = -0.2     # 20% below average = declining
    
    # Momentum score adjustments
    ACCELERATING_BONUS = 15        # +15 points for accelerating papers
    STABLE_BONUS = 0               # No change for stable
    DECLINING_PENALTY = -10        # -10 for declining papers
    
    def __init__(
        self,
        recent_window_months: int = 6,
        min_citations_for_trend: int = 5,
    ):
        """
        Initialize the calculator.
        
        Args:
            recent_window_months: Window for "recent" citations
            min_citations_for_trend: Minimum citations needed to compute trend
        """
        self.recent_window_months = recent_window_months
        self.min_citations_for_trend = min_citations_for_trend
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_citation_velocity(
        self,
        paper,
        citation_history: Optional[Dict[int, int]] = None,
    ) -> CitationVelocity:
        """
        Compute citation velocity for a paper.
        
        Args:
            paper: Paper metadata with citation_count and year
            citation_history: Optional dict of {year: citation_count}
            
        Returns:
            CitationVelocity with trend analysis
        """
        result = CitationVelocity(
            window_months=self.recent_window_months,
            trend=CitationTrend.STABLE,
        )
        
        total_citations = getattr(paper, 'citation_count', 0) or 0
        paper_year = getattr(paper, 'year', None)
        
        if total_citations < self.min_citations_for_trend:
            result.recent_citations = total_citations
            return result
        
        # If we have citation history, use it
        if citation_history:
            return self._compute_from_history(paper, citation_history)
        
        # Otherwise, estimate from total and year
        return self._estimate_velocity(paper, total_citations, paper_year)
    
    def _compute_from_history(
        self,
        paper,
        history: Dict[int, int],
    ) -> CitationVelocity:
        """Compute velocity from citation history."""
        current_year = datetime.now().year
        
        # Get recent citations (this year + fraction of previous)
        recent = history.get(current_year, 0)
        
        # Adjust for window (e.g., 6 months = half of this year)
        months_elapsed = datetime.now().month
        if months_elapsed < self.recent_window_months:
            # Include some from previous year
            prev_year_fraction = (self.recent_window_months - months_elapsed) / 12
            recent += int(history.get(current_year - 1, 0) * prev_year_fraction)
        
        # Compute annual average
        years = sorted(history.keys())
        if len(years) >= 2:
            total = sum(history.values())
            year_span = max(years) - min(years) + 1
            annual_avg = total / year_span
        else:
            annual_avg = sum(history.values())
        
        # Compute growth rate
        if annual_avg > 0:
            expected_recent = annual_avg * (self.recent_window_months / 12)
            growth_rate = (recent - expected_recent) / expected_recent if expected_recent > 0 else 0
        else:
            growth_rate = 0
        
        # Classify trend
        if growth_rate > self.ACCELERATING_THRESHOLD:
            trend = CitationTrend.ACCELERATING
        elif growth_rate < self.DECLINING_THRESHOLD:
            trend = CitationTrend.DECLINING
        else:
            trend = CitationTrend.STABLE
        
        return CitationVelocity(
            recent_citations=recent,
            annual_average=round(annual_avg, 1),
            growth_rate=round(growth_rate * 100, 1),
            trend=trend,
            window_months=self.recent_window_months,
        )
    
    def _estimate_velocity(
        self,
        paper,
        total_citations: int,
        paper_year: Optional[int],
    ) -> CitationVelocity:
        """Estimate velocity when history is unavailable."""
        current_year = datetime.now().year
        
        if not paper_year:
            paper_year = current_year - 2  # Default assumption
        
        years_since_pub = max(1, current_year - int(paper_year))
        
        # Compute annual average
        annual_avg = total_citations / years_since_pub
        
        # Estimate recent citations
        # For recent papers (< 2 years), assume recent = annual
        # For older papers, estimate decay
        if years_since_pub <= 2:
            recent_estimate = annual_avg * (self.recent_window_months / 12)
            trend = CitationTrend.STABLE
        else:
            # Assume some decay for older papers
            decay_factor = 0.7 ** (years_since_pub - 2)
            recent_estimate = annual_avg * decay_factor * (self.recent_window_months / 12)
            
            # Papers with high citations but old are likely declining
            if decay_factor < 0.5:
                trend = CitationTrend.DECLINING
            else:
                trend = CitationTrend.STABLE
        
        return CitationVelocity(
            recent_citations=int(recent_estimate),
            annual_average=round(annual_avg, 1),
            growth_rate=0.0,  # Unknown without history
            trend=trend,
            window_months=self.recent_window_months,
        )
    
    def compute_momentum_score(self, velocity: CitationVelocity) -> float:
        """
        Convert velocity to a momentum score adjustment.
        
        Args:
            velocity: CitationVelocity result
            
        Returns:
            Score adjustment (-10 to +15)
        """
        if velocity.trend == CitationTrend.ACCELERATING:
            return self.ACCELERATING_BONUS
        elif velocity.trend == CitationTrend.DECLINING:
            return self.DECLINING_PENALTY
        else:
            return self.STABLE_BONUS
    
    def get_trend_explanation(self, velocity: CitationVelocity) -> str:
        """Generate human-readable explanation of the trend."""
        if velocity.trend == CitationTrend.ACCELERATING:
            return (
                f"📈 加速增长: 近{velocity.window_months}个月获得{velocity.recent_citations}次引用, "
                f"年均{velocity.annual_average}, 增长率{velocity.growth_rate}%"
            )
        elif velocity.trend == CitationTrend.DECLINING:
            return (
                f"📉 引用下降: 近期引用{velocity.recent_citations}次, "
                f"低于年均{velocity.annual_average}"
            )
        else:
            return (
                f"➡️ 引用稳定: 年均{velocity.annual_average}次引用"
            )
