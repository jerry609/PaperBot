from __future__ import annotations

from typing import Any, Dict

from paperbot.application.services.llm_service import LLMService, get_llm_service


class RelevanceAssessor:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or get_llm_service()

    def assess(self, *, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        return self.llm_service.assess_relevance(paper=paper, query=query)
