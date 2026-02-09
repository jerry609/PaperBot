from __future__ import annotations

from typing import Any, Dict

from paperbot.application.services.llm_service import LLMService, get_llm_service


class PaperSummarizer:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or get_llm_service()

    def summarize_item(self, item: Dict[str, Any]) -> str:
        return self.llm_service.summarize_paper(
            title=item.get("title") or "",
            abstract=item.get("snippet") or item.get("abstract") or "",
        )
