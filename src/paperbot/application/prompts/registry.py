from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from paperbot.application.prompts.paper_analysis import (
    PAPER_SUMMARY_SYSTEM,
    PAPER_SUMMARY_USER,
    RELEVANCE_ASSESS_SYSTEM,
    RELEVANCE_ASSESS_USER,
)
from paperbot.application.prompts.trend_detection import (
    DAILY_INSIGHT_SYSTEM,
    DAILY_INSIGHT_USER,
    TREND_ANALYSIS_SYSTEM,
    TREND_ANALYSIS_USER,
)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user: str


class PromptRegistry:
    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {
            "paper_summary": PromptTemplate(
                name="paper_summary",
                system=PAPER_SUMMARY_SYSTEM,
                user=PAPER_SUMMARY_USER,
            ),
            "trend_analysis": PromptTemplate(
                name="trend_analysis",
                system=TREND_ANALYSIS_SYSTEM,
                user=TREND_ANALYSIS_USER,
            ),
            "relevance_assess": PromptTemplate(
                name="relevance_assess",
                system=RELEVANCE_ASSESS_SYSTEM,
                user=RELEVANCE_ASSESS_USER,
            ),
            "daily_insight": PromptTemplate(
                name="daily_insight",
                system=DAILY_INSIGHT_SYSTEM,
                user=DAILY_INSIGHT_USER,
            ),
        }

    def get(self, name: str) -> PromptTemplate:
        key = (name or "").strip().lower()
        if key not in self._templates:
            raise KeyError(f"Unknown prompt template: {name}")
        return self._templates[key]

    def list(self):
        return sorted(self._templates.keys())
