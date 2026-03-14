"""EnrichmentPipeline — Chain of Responsibility for paper enrichment.

Each step processes a paper and can add judge scores, summaries,
or post-filter flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentContext:
    """Shared context passed through the pipeline."""

    query: str = ""
    user_id: Optional[str] = None
    track_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EnrichmentStep(Protocol):
    """Single enrichment step."""

    async def process(self, paper: Dict[str, Any], context: EnrichmentContext) -> None:
        """Mutate *paper* dict in-place with enrichment data."""
        ...


class EnrichmentPipeline:
    """Runs a chain of EnrichmentStep instances over a list of papers."""

    def __init__(self, steps: Optional[List[EnrichmentStep]] = None):
        self.steps: List[EnrichmentStep] = [s for s in (steps or []) if s is not None]

    async def run(
        self,
        papers: List[Dict[str, Any]],
        context: Optional[EnrichmentContext] = None,
    ) -> None:
        ctx = context or EnrichmentContext()
        for paper in papers:
            for step in self.steps:
                try:
                    await step.process(paper, ctx)
                except Exception as e:
                    title = str(paper.get("title", ""))[:60]
                    step_name = type(step).__name__
                    logger.warning(f"Enrichment step {step_name} failed for {title}: {e}")


class LLMEnrichmentStep:
    """Attach LLM summary/relevance features to selected papers."""

    def __init__(self, *, llm_service=None, features: Optional[List[str]] = None):
        from paperbot.application.services.llm_service import get_llm_service

        self._llm = llm_service or get_llm_service()
        self._features = set(features or ["summary"])

    async def process(self, paper: Dict[str, Any], context: EnrichmentContext) -> None:
        target_ids = context.extra.get("llm_target_ids")
        if isinstance(target_ids, set) and id(paper) not in target_ids:
            return

        title = str(paper.get("title") or "")
        abstract = str(paper.get("snippet") or paper.get("abstract") or "")

        if "summary" in self._features:
            paper["ai_summary"] = self._llm.summarize_paper(title=title, abstract=abstract)

        if "relevance" in self._features:
            query = str(context.extra.get("query_for_relevance") or context.query or "")
            paper["relevance"] = self._llm.assess_relevance(paper=paper, query=query)


class JudgeStep:
    """Attach judge scores to selected papers."""

    def __init__(self, *, judge=None, n_runs: int = 1):
        if judge is None:
            from paperbot.application.services.llm_service import get_llm_service
            from paperbot.application.workflows.analysis.paper_judge import PaperJudge

            judge = PaperJudge(llm_service=get_llm_service())
        self._judge = judge
        self._n_runs = max(1, int(n_runs))

    async def process(self, paper: Dict[str, Any], context: EnrichmentContext) -> None:
        target_ids = context.extra.get("judge_target_ids")
        if isinstance(target_ids, set) and id(paper) not in target_ids:
            return

        query_map = context.extra.get("paper_query_map") or {}
        query = str(query_map.get(id(paper)) or context.query or "")

        if self._n_runs > 1:
            judgment = self._judge.judge_with_calibration(
                paper=paper,
                query=query,
                n_runs=self._n_runs,
            )
        else:
            judgment = self._judge.judge_single(paper=paper, query=query)
        paper["judge"] = judgment.to_dict()


class FilterStep:
    """Mark papers as filtered when recommendation is not in keep-set."""

    def __init__(self, keep: Optional[set[str]] = None):
        self._keep = keep or {"must_read", "worth_reading"}

    async def process(self, paper: Dict[str, Any], context: EnrichmentContext) -> None:
        judge = paper.get("judge")
        if not isinstance(judge, dict):
            return
        rec = str(judge.get("recommendation") or "").strip().lower()
        if rec and rec not in self._keep:
            paper["_filtered_out"] = True


class StructuredCardStep:
    """Extract structured card (method/dataset/conclusion/limitations) via LLM."""

    def __init__(self, *, llm_service=None):
        from paperbot.application.services.llm_service import get_llm_service

        self._llm = llm_service or get_llm_service()

    async def process(self, paper: Dict[str, Any], context: EnrichmentContext) -> None:
        if paper.get("structured_card"):
            return

        title = str(paper.get("title") or "")
        abstract = str(paper.get("snippet") or paper.get("abstract") or "")
        if not abstract:
            return

        card = self._llm.extract_structured_card(title=title, abstract=abstract)
        paper["structured_card"] = card
