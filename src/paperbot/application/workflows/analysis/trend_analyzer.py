from __future__ import annotations

from typing import Any, Dict, Sequence

from paperbot.application.services.llm_service import LLMService, get_llm_service


class TrendAnalyzer:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or get_llm_service()

    def analyze(self, *, topic: str, items: Sequence[Dict[str, Any]]) -> str:
        return self.llm_service.analyze_trends(topic=topic, papers=items)
